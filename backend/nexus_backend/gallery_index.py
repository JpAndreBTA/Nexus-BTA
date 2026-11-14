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
    filename: str = ""


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


VALID_MEDIA_TYPES = frozenset({"all", "image", "video", "audio", "3d"})
VALID_SORT_ORDERS = frozenset({"newest", "oldest", "name", "type"})


def _query_signature(query: str, media_type: str, sort_order: str) -> str:
    raw = json.dumps([query.casefold(), media_type, sort_order], separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("ascii")).hexdigest()[:16]


def _encode_cursor(item: IndexedMedia, signature: str) -> str:
    raw = json.dumps(
        [2, signature, item.modified_ns, item.relative_path, item.filename, item.media_type],
        ensure_ascii=True, separators=(",", ":")
    ).encode("ascii")
    return base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=")


def _decode_cursor(value: str | None, signature: str) -> tuple[int, str, str, str] | None:
    if not value:
        return None
    if len(value) > 768 or not all(c.isascii() and (c.isalnum() or c in "-_") for c in value):
        raise ValueError("Invalid gallery cursor.")
    try:
        data = json.loads(base64.urlsafe_b64decode(value + "=" * (-len(value) % 4)))
    except (ValueError, UnicodeError) as exc:
        raise ValueError("Invalid gallery cursor.") from exc
    if (
        isinstance(data, list) and len(data) == 2 and
        type(data[0]) is int and isinstance(data[1], str)
    ):
        # Existing cursors are accepted for the unchanged default newest view.
        if signature != _query_signature("", "all", "newest"):
            raise ValueError("Cursor does not match gallery filters.")
        return data[0], data[1], "", ""
    if (
        not isinstance(data, list) or len(data) != 6 or data[0] != 2 or
        data[1] != signature or type(data[2]) is not int or
        not all(isinstance(x, str) for x in data[3:])
        or len(data[3]) > 640 or "\\x00" in data[3]
    ):
        raise ValueError("Invalid gallery cursor.")
    return data[2], data[3], data[4], data[5]


def _escape_like(text: str) -> str:
    return text.replace("!", "!!").replace("%", "!%").replace("_", "!_")



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
                    media_type TEXT NOT NULL,
                filename TEXT NOT NULL DEFAULT ''
                )"""
            )
            # Upgrade databases written by M2.2 in place, preserving the existing index.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(media)")}
            if "filename" not in columns:
                connection.execute("ALTER TABLE media ADD COLUMN filename TEXT NOT NULL DEFAULT ''")
            connection.execute("CREATE INDEX IF NOT EXISTS media_recent ON media (modified_ns DESC, relative_path DESC)")
            connection.execute("CREATE INDEX IF NOT EXISTS media_name ON media (filename COLLATE NOCASE, relative_path)")
            connection.execute("CREATE INDEX IF NOT EXISTS media_type_recent ON media (media_type, modified_ns DESC, relative_path DESC)")
            with connection:
                yield connection
        finally:
            connection.close()

    def _scan(self, connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TEMP TABLE IF NOT EXISTS seen (relative_path TEXT PRIMARY KEY)")
        connection.execute("DELETE FROM seen")
        batch: list[tuple[str, int, int, str, str]] = []

        def flush() -> None:
            if not batch:
                return
            connection.executemany("INSERT OR IGNORE INTO seen(relative_path) VALUES (?)", ((x[0],) for x in batch))
            connection.executemany(
                """INSERT INTO media(relative_path, modified_ns, size_bytes, media_type, filename)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(relative_path) DO UPDATE SET
                    modified_ns=excluded.modified_ns,
                    size_bytes=excluded.size_bytes,
                    media_type=excluded.media_type,
                    filename=excluded.filename
                WHERE media.modified_ns != excluded.modified_ns
                   OR media.size_bytes != excluded.size_bytes
                   OR media.media_type != excluded.media_type
                   OR media.filename != excluded.filename""", batch,
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
                    batch.append((relative.as_posix(), stat.st_mtime_ns, stat.st_size, _media_type(suffix), filename))
                    if len(batch) >= 256:
                        flush()
        flush()
        connection.execute("DELETE FROM media WHERE relative_path NOT IN (SELECT relative_path FROM seen)")
        connection.execute("DELETE FROM seen")

    def page(
        self,
        limit: int = 48,
        cursor: str | None = None,
        force_refresh: bool = False,
        query: str = "",
        media_type: str = "all",
        sort_order: str = "newest",
    ) -> IndexedPage:
        if not 1 <= limit <= 100:
            raise ValueError("Gallery page size must be between 1 and 100.")
        query = query.strip()
        if len(query) > 120 or "\\x00" in query:
            raise ValueError("Invalid gallery search.")
        if media_type not in VALID_MEDIA_TYPES or sort_order not in VALID_SORT_ORDERS:
            raise ValueError("Invalid gallery filter or sort.")
        signature = _query_signature(query, media_type, sort_order)
        marker = _decode_cursor(cursor, signature)
        conditions: list[str] = []
        params: list[object] = []
        if query:
            conditions.append("relative_path LIKE ? ESCAPE '!' COLLATE NOCASE")
            params.append(f"%{_escape_like(query)}%")
        if media_type != "all":
            conditions.append("media_type = ?")
            params.append(media_type)
        base_where = " WHERE " + " AND ".join(conditions) if conditions else ""

        with self._lock:
            with self._connect() as connection:
                if force_refresh or not self._last_scan or time.monotonic() - self._last_scan >= self.refresh_seconds:
                    self._scan(connection)
                    connection.commit()
                    self._last_scan = time.monotonic()
                total = int(connection.execute("SELECT COUNT(*) FROM media" + base_where, params).fetchone()[0])

                ordered = {
                    "newest": "modified_ns DESC, relative_path DESC",
                    "oldest": "modified_ns ASC, relative_path ASC",
                    "name": "filename COLLATE NOCASE ASC, relative_path ASC",
                    "type": "media_type ASC, modified_ns DESC, relative_path DESC",
                }[sort_order]
                page_params = list(params)
                page_conditions = list(conditions)
                if marker:
                    modified_ns, path, filename, kind = marker
                    if sort_order == "newest":
                        page_conditions.append("(modified_ns < ? OR (modified_ns = ? AND relative_path < ?))")
                        page_params.extend((modified_ns, modified_ns, path))
                    elif sort_order == "oldest":
                        page_conditions.append("(modified_ns > ? OR (modified_ns = ? AND relative_path > ?))")
                        page_params.extend((modified_ns, modified_ns, path))
                    elif sort_order == "name":
                        page_conditions.append(
                            "(filename COLLATE NOCASE > ? OR (filename COLLATE NOCASE = ? AND relative_path > ?))"
                        )
                        page_params.extend((filename, filename, path))
                    else:
                        page_conditions.append(
                            "(media_type > ? OR (media_type = ? AND "
                            "(modified_ns < ? OR (modified_ns = ? AND relative_path < ?))))"
                        )
                        page_params.extend((kind, kind, modified_ns, modified_ns, path))
                clause = " WHERE " + " AND ".join(page_conditions) if page_conditions else ""
                rows = connection.execute(
                    "SELECT relative_path, modified_ns, size_bytes, media_type, filename FROM media" +
                    clause + " ORDER BY " + ordered + " LIMIT ?",
                    [*page_params, limit + 1],
                ).fetchall()
                results = [IndexedMedia(*row) for row in rows[:limit]]
                next_cursor = _encode_cursor(results[-1], signature) if len(rows) > limit and results else None
                return IndexedPage(records=results, next_cursor=next_cursor, total=total)
