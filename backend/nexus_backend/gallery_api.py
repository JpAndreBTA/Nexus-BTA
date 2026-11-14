"""Standalone indexed-gallery API router, independent of the monolithic main.py.

No GPU, model registry, or filesystem modifications are needed to test this.
"""
from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse

from .gallery_index import GalleryIndex, IndexedMedia, default_gallery_db_path, IMAGE_SUFFIXES
from .gallery_thumbnail import GalleryThumbnailCache


def make_gallery_router(
    output_dir: Callable[[], Path],
    read_metadata: Callable[[Path], dict[str, Any]],
) -> APIRouter:
    router = APIRouter()
    _gallery_page_index: GalleryIndex | None = None
    _gallery_index_lock = threading.Lock()
    _gallery_thumbnail_cache: GalleryThumbnailCache | None = None

    def _get_gallery_index() -> GalleryIndex:
        """Keep the indexed route independent from the legacy 200-item response."""
        nonlocal _gallery_page_index, _gallery_thumbnail_cache
        root = output_dir().resolve()
        with _gallery_index_lock:
            if _gallery_page_index is None or _gallery_page_index.root != root:
                _gallery_page_index = GalleryIndex(root, default_gallery_db_path(root))
                _gallery_thumbnail_cache = GalleryThumbnailCache(root)
            return _gallery_page_index


    def _indexed_gallery_item(indexed: IndexedMedia, root: Path) -> dict[str, Any] | None:
        """Serialize one indexed output without trusting a path stored in SQLite."""
        path = root / indexed.relative_path
        try:
            if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root) or not path.is_file():
                return None
            modified = path.stat().st_mtime
        except OSError:
            return None
        relative = path.relative_to(root).as_posix()
        url_path = quote(relative, safe="/")
        metadata = read_metadata(path)
        return {
            "title": path.name,
            "filename": path.name,
            # The new paginated API intentionally does not expose Windows absolute paths.
            "path": relative,
            "relative_path": relative,
            "folder": path.parent.relative_to(root).as_posix() if path.parent != root else "",
            "image": f"/outputs/{url_path}",
            "thumb": f"/api/gallery/thumb/{url_path}?size=320" if path.suffix.lower() in IMAGE_SUFFIXES else f"/outputs/{url_path}",
            "media_type": indexed.media_type,
            "prompt": metadata.get("prompt", ""),
            "negative": metadata.get("negative", ""),
            "model": metadata.get("model", ""),
            "seed": str(metadata.get("seed", "")),
            "steps": str(metadata.get("steps", "")),
            "cfg": str(metadata.get("cfg", "")),
            "sampler": metadata.get("sampler", ""),
            "scheduler": metadata.get("scheduler", ""),
            "preset": metadata.get("preset", ""),
            "activity": metadata.get("activity", ""),
            "width": metadata.get("width", ""),
            "height": metadata.get("height", ""),
            "workflow_id": metadata.get("workflow_id", ""),
            "loras": metadata.get("loras", []),
            "distilled_loras": metadata.get("distilled_loras", []),
            "controlnet": metadata.get("controlnet", {}),
            "vae": metadata.get("vae", ""),
            "text_encoder": metadata.get("text_encoder", ""),
            "video": metadata.get("video", {}),
            "director": metadata.get("director", {}),
            "metadata": metadata,
            "modified": modified,
        }



    @router.get("/api/gallery/thumb/{relative_path:path}")
    async def image_thumbnail(
        relative_path: str,
        size: int = Query(default=320, ge=64, le=512),
    ) -> FileResponse:
        index = _get_gallery_index()
        with _gallery_index_lock:
            cache = _gallery_thumbnail_cache
        if cache is None:
            raise HTTPException(status_code=503, detail="Gallery cache unavailable.")
        try:
            thumbnail_path = await asyncio.to_thread(cache.thumbnail, relative_path, size)
        except ValueError as exc:
            # A malformed file never leaks backend details/absolute paths to clients.
            raise HTTPException(status_code=404, detail="Thumbnail not available.") from exc
        return FileResponse(
            thumbnail_path, media_type="image/webp",
            headers={"Cache-Control": "private, max-age=3600", "X-Content-Type-Options": "nosniff"},
        )

    @router.get("/api/gallery/page")
    async def gallery_page(
        limit: int = Query(default=48, ge=1, le=100),
        cursor: str | None = Query(default=None, max_length=512),
        refresh: bool = Query(default=False),
        q: str = Query(default="", max_length=120),
        media: str = Query(default="all", max_length=16),
        sort: str = Query(default="newest", max_length=16),
    ) -> dict[str, Any]:
        """Keyset pagination for React Gallery; legacy /api/gallery stays unchanged.

        Disk scanning is executed outside the event loop and throttled to 10s.
        A non-existent output is an empty result. Stale/deleted files are skipped.
        """
        index = _get_gallery_index()
        try:
            page = await asyncio.to_thread(index.page, limit, cursor, refresh, q, media, sort)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        # Metadata decoding is bounded by the requested page size, and must not
        # block FastAPI's event loop when a large PNG has embedded metadata.
        def serialize_page() -> list[dict[str, Any]]:
            items: list[dict[str, Any]] = []
            for entry in page.records:
                record = _indexed_gallery_item(entry, index.root)
                if record is not None:
                    items.append(record)
            return items

        items = await asyncio.to_thread(serialize_page)
        return {"items": items, "total": page.total, "next_cursor": page.next_cursor}

    return router
