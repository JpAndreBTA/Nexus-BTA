"""Persistent, bounded-memory, read-only output gallery index.

The SQLite database stores relative file paths and metadata, never media bytes.
Scanning is incremental by file metadata and throttled; watchers are a future step.
The legacy /api/gallery API is intentionally left unchanged.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp", ".gif"})
VIDEO_SUFFIXES = frozenset({".mp4", ".webm", ".mkv", ".mov", ".avi"})
AUDIO_SUFFIXES = frozenset({".mp3", ".wav", ".flac", ".opus", ".ogg", ".m4a"})
MODEL_SUFFIXES = frozenset({".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz", ".spz", ".ksplat"})
MEDIA_SUFFIXES = IMAGE_SUFFIXES | VIDEO_SUFFIXES | AUDIO_SUFFIXES | MODEL_SUFFIXES


@dataclass(frozen=True)
class IndexedMedia:
    relative_path: str
    modified_ns: int
    size_bytes: int
    media_type: str


@dataclass(frozen=True)
class IndexedPage:
    records: list[IndexedMedia]
    next_cursor: str | None
    total: int


def default_gallery_db_path(output_root: Path) -> Path:
    """Keep SQLite outside /outputs and outside any HTTP static mount."""
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("XDG_CACHE_HOME")
    cache_dir = (Path(base) if base else Path.home() / ".cache") / "Nexus-BTA"
    digest = hashlib.sha256(os.fsencode(str(output_root.resolve()))).hexdigest()[:20]
    return cache_dir / f"gallery-{digest}.sqlite3"


def _encode_cursor(item: IndexedMedia) -> str:
    raw = json.dumps([item.modified_ns, item.relative_path], ensure_ascii=True, separators=(",", ":")).encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None) -> tuple[int, str] | None:
    if not value:
        return None
    if len(value) > 512 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in value):
        raise ValueError("Invalid gallery cursor.")
    try:
        data = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid gallery cursor.") from exc
    if (
        not isinstance(data, list) or len(data) != 2
        or type(data[0]) is not int or not isinstance(data[1], str)
        or len(data[1]) > 400 or "\x00" in data[1]
    ):
        raise ValueError("Invalid gallery cursor.")
    return data[0], data[1]


def _media_type(suffix: str) -> str:
    if suffix in MODEL_SUFFIXES:
        return "3d"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    return "image"


class GalleryIndex:
    """Thread-safe SQLite index, with keyset pagination and TTL rescanning.

    A scan is O(number of files) and writes in batches, but normal paginated
    reads are indexed and do not enumerate the filesystem.
    """

    def __init__(self, output_root: Path, db_path: Path, refresh_seconds: float = 10.0):
        self.root = output_root.resolve()
        self.db_path = db_path.resolve()
        if self.db_path.is_relative_to(self.root):
            raise ValueError("Gallery database cannot be inside the exposed output root.")
        self.refresh_seconds = max(0, refresh_seconds)
        self._last_scan = 0.0
        self._lock = threading.RLock()

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """sqlite3.Connection.__exit__ only commits; it does NOT close the handle."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.db_path, timeout=10.0)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS media (
                    relative_path TEXT PRIMARY KEY,
                    modified_ns INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    media_type TEXT NOT NULL
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS media_recent ON media (modified_ns DESC, relative_path DESC)")
            with connection:
                yield connection
        finally:
            connection.close()

    def _scan(self, connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TEMP TABLE IF NOT EXISTS seen (relative_path TEXT PRIMARY KEY)")
        connection.execute("DELETE FROM seen")
        batch: list[tuple[str, int, int, str]] = []

        def flush() -> None:
            if not batch:
                return
            connection.executemany("INSERT OR IGNORE INTO seen(relative_path) VALUES (?)", ((x[0],) for x in batch))
            connection.executemany(
                """INSERT INTO media(relative_path, modified_ns, size_bytes, media_type)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    modified_ns=excluded.modified_ns,
                    size_bytes=excluded.size_bytes,
                    media_type=excluded.media_type
                WHERE media.modified_ns != excluded.modified_ns
                   OR media.size_bytes != excluded.size_bytes
                   OR media.media_type != excluded.media_type""", batch,
            )
            batch.clear()

        def raise_walk_error(error: OSError) -> None:
            raise error

        if self.root.is_dir():
            for directory, folders, filenames in os.walk(self.root, followlinks=False, onerror=raise_walk_error):
                parent = Path(directory)
                folders[:] = [
                    name for name in folders
                    if not (parent / name).is_symlink() and (parent / name).resolve().is_relative_to(self.root)
                ]
                for filename in filenames:
                    path = parent / filename
                    suffix = path.suffix.lower()
                    if suffix not in MEDIA_SUFFIXES:
                        continue
                    try:
                        if path.is_symlink() or not path.resolve(strict=True).is_relative_to(self.root):
                            continue
                        relative = path.relative_to(self.root)
                        if suffix in IMAGE_SUFFIXES and relative.parts and relative.parts[0].casefold() == "video":
                            continue
                        stat = path.stat()
                        if not path.is_file():
                            continue
                    except OSError:
                        # Media may be deleted while a generation is finishing.
                        continue
                    batch.append((relative.as_posix(), stat.st_mtime_ns, stat.st_size, _media_type(suffix)))
                    if len(batch) >= 256:
                        flush()
        flush()
        connection.execute("DELETE FROM media WHERE relative_path NOT IN (SELECT relative_path FROM seen)")
        connection.execute("DELETE FROM seen")

    def page(self, limit: int = 48, cursor: str | None = None, force_refresh: bool = False) -> IndexedPage:
        if not 1 <= limit <= 100:
            raise ValueError("Gallery page size must be between 1 and 100.")
        marker = _decode_cursor(cursor)
        with self._lock:
            with self._connect() as connection:
                if force_refresh or not self._last_scan or time.monotonic() - self._last_scan >= self.refresh_seconds:
                    self._scan(connection)
                    connection.commit()
                    self._last_scan = time.monotonic()
                total = int(connection.execute("SELECT COUNT(*) FROM media").fetchone()[0])
                if marker:
                    rows = connection.execute(
                        """SELECT relative_path, modified_ns, size_bytes, media_type
                           FROM media
                           WHERE modified_ns < ? OR (modified_ns = ? AND relative_path < ?)
                           ORDER BY modified_ns DESC, relative_path DESC LIMIT ?""",
                        (marker[0], marker[0], marker[1], limit + 1),
                    ).fetchall()
                else:
                    rows = connection.execute(
                        """SELECT relative_path, modified_ns, size_bytes, media_type
                           FROM media ORDER BY modified_ns DESC, relative_path DESC LIMIT ?""",
                        (limit + 1,),
                    ).fetchall()
                results = [IndexedMedia(*row) for row in rows[:limit]]
                next_cursor = _encode_cursor(results[-1]) if len(rows) > limit and results else None
                return IndexedPage(records=results, next_cursor=next_cursor, total=total)
