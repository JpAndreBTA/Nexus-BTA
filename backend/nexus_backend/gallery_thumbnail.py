"""On-demand thumbnail cache for local gallery images.

Never copies original metadata, scans unrelated paths, or writes under /outputs.
The cache is bounded and keyed by source file identity (relative path, mtime, size).
No previews are generated eagerly while indexing the output folder.
"""
from __future__ import annotations

import hashlib
import io
import os
import tempfile
import threading
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError

from .gallery_index import IMAGE_SUFFIXES, default_gallery_db_path

MAX_SOURCE_BYTES = 128 * 1024 * 1024
MAX_SOURCE_PIXELS = 64_000_000
MAX_CACHE_BYTES = 384 * 1024 * 1024
MAX_EDGE = 512


def safe_image_path(root: Path, relative_path: str) -> Path:
    """Reject traversal and symlink escape even when a URL was manually crafted."""
    if (
        not relative_path or len(relative_path) > 1024 or
        relative_path.startswith(("/", "\\\\")) or "\\\\" in relative_path or "\\x00" in relative_path or
        any(p in ("", ".", "..") for p in relative_path.split("/"))
    ):
        raise ValueError("Invalid thumbnail path.")
    candidate = root / relative_path
    if candidate.suffix.lower() not in IMAGE_SUFFIXES or candidate.is_symlink():
        raise ValueError("Unsupported thumbnail input.")
    try:
        resolved = candidate.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("Thumbnail source is outside the gallery.")
    except OSError as exc:
        raise ValueError("Thumbnail image not found.") from exc
    return resolved


class GalleryThumbnailCache:
    def __init__(self, root: Path, *, cache_root: Path | None = None):
        self.root = root.resolve()
        namespace = hashlib.sha256(os.fsencode(str(self.root))).hexdigest()[:20]
        self.cache_root = (cache_root or (default_gallery_db_path(self.root).parent / "gallery-thumbs" / namespace)).resolve()
        if self.cache_root.is_relative_to(self.root):
            raise ValueError("Thumbnail cache must be outside public outputs.")
        self._lock = threading.RLock()

    def _trim_cache(self) -> None:
        """Best-effort LRU-by-mtime eviction, at most 384 MiB on disk."""
        files: list[tuple[float, int, Path]] = []
        total = 0
        for path in self.cache_root.glob("*.webp"):
            try:
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            files.append((stat.st_mtime, stat.st_size, path))
        if total <= MAX_CACHE_BYTES:
            return
        for _, size, path in sorted(files):
            try:
                path.unlink()
                total -= size
            except OSError:
                pass
            if total <= MAX_CACHE_BYTES:
                break

    def thumbnail(self, relative_path: str, size: int = 320) -> Path:
        if not 64 <= size <= MAX_EDGE:
            raise ValueError("Thumbnail size must be between 64 and 512.")
        source = safe_image_path(self.root, relative_path)
        stat = source.stat()
        if stat.st_size > MAX_SOURCE_BYTES:
            raise ValueError("Image exceeds thumbnail processing size limit.")
        key = hashlib.sha256(
            f"{relative_path}\\0{stat.st_mtime_ns}\\0{stat.st_size}\\0{size}".encode("utf-8")
        ).hexdigest()
        output = self.cache_root / f"{key}.webp"
        with self._lock:
            if output.is_file():
                return output
            self.cache_root.mkdir(parents=True, exist_ok=True)
            # Limit source pixels *before* converting/decoding full RGB/RGBA surfaces.
            try:
                with Image.open(source) as opened:
                    if opened.width * opened.height > MAX_SOURCE_PIXELS:
                        raise ValueError("Image exceeds thumbnail pixel budget.")
                    opened.draft("RGB", (size * 2, size * 2))
                    opened.thumbnail((size, size), Image.Resampling.LANCZOS)
                    upright = ImageOps.exif_transpose(opened)
                    upright.thumbnail((size, size), Image.Resampling.LANCZOS)
                    if upright.mode not in ("RGB", "RGBA"):
                        upright = upright.convert("RGBA" if "A" in upright.getbands() else "RGB")
                    data = io.BytesIO()
                    upright.save(data, format="WEBP", quality=74, method=4)
            except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
                raise ValueError("Unable to decode thumbnail source.") from exc
            fd, scratch = tempfile.mkstemp(prefix=".tmp-gallery-", suffix=".webp", dir=self.cache_root)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(data.getvalue())
                os.replace(scratch, output)
            finally:
                if os.path.exists(scratch):
                    os.unlink(scratch)
            self._trim_cache()
            return output
