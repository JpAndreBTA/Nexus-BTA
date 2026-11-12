"""Memory-bounded selection of recent gallery files.

A full filesystem index (SQLite + watcher) is planned for a later milestone;
this module replaces O(N log N) sorting of all entries with O(N log limit)
selection and O(limit) memory without changing the API response contract.
"""
from __future__ import annotations

import heapq
from itertools import count
from pathlib import Path


def recent_media_files(
    root: Path, extensions: set[str], limit: int = 200, skip_video_images: bool = False
) -> list[tuple[Path, float]]:
    if limit <= 0 or not root.is_dir():
        return []
    recent: list[tuple[float, int, Path]] = []
    sequence = count()
    for path in root.rglob("*"):
        try:
            suffix = path.suffix.lower()
            if path.is_symlink() or suffix not in extensions or not path.is_file():
                continue
            if skip_video_images and suffix in {".png", ".jpg", ".jpeg", ".webp", ".gif"}:
                if path.relative_to(root).as_posix().startswith("video/"):
                    continue
            modified = path.stat().st_mtime
        except OSError:
            # Outputs can be deleted while a generation is finishing.
            continue
        entry = (modified, next(sequence), path)
        if len(recent) < limit:
            heapq.heappush(recent, entry)
        elif modified > recent[0][0]:
            heapq.heapreplace(recent, entry)
    return [(path, modified) for modified, _, path in sorted(recent, reverse=True)]
