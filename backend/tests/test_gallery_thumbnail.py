"""On-demand Gallery image thumbnail and traversal regression tests."""
from __future__ import annotations

import tempfile
import unittest
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

from nexus_backend.gallery_api import make_gallery_router
from nexus_backend.gallery_thumbnail import GalleryThumbnailCache, safe_image_path


class GalleryThumbnailTests(unittest.TestCase):
    def test_image_processing_caching_and_page_endpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            original = root / "painting.png"
            Image.new("RGBA", (1900, 970), (25, 70, 130, 200)).save(original)
            app = FastAPI()
            app.include_router(make_gallery_router(lambda: root, lambda _: {}))
            client = TestClient(app)
            page = client.get("/api/gallery/page?limit=2").json()
            self.assertEqual(page["total"], 1)
            self.assertEqual(page["items"][0]["thumb"], "/api/gallery/thumb/painting.png?size=320")
            response = client.get(page["items"][0]["thumb"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["content-type"], "image/webp")
            self.assertLess(len(response.content), original.stat().st_size + 4096)
            with Image.open(BytesIO(response.content)) as thumb:
                self.assertLessEqual(max(thumb.size), 320)
            self.assertEqual(response.content, client.get(page["items"][0]["thumb"]).content)

    def test_traversal_symlink_and_wrong_media_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            (root / "valid.png").write_bytes(b"not an image")
            (root / "clip.mp4").write_bytes(b"movie")
            for value in ("../secret.png", r"nested\\a.png", "/etc/passwd", "clip.mp4"):
                with self.assertRaises(ValueError):
                    safe_image_path(root, value)
            index = GalleryThumbnailCache(root, cache_root=Path(tmp) / "private-cache")
            with self.assertRaises(ValueError):
                index.thumbnail("valid.png")
            with self.assertRaises(ValueError):
                index.thumbnail("valid.png", size=4096)
            outside = Path(tmp) / "secret.png"
            outside.write_bytes(b"private")
            try:
                (root / "link.png").symlink_to(outside)
            except OSError:
                pass
            else:
                with self.assertRaises(ValueError):
                    safe_image_path(root, "link.png")
            self.assertFalse(index.cache_root.is_relative_to(root))

    def test_thumbnail_cache_never_trusts_symlinked_cache_entry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            Image.new("RGB", (96, 80), (12, 100, 200)).save(root / "cover.png")
            cache = GalleryThumbnailCache(root, cache_root=Path(tmp) / "cache")
            rendered = cache.thumbnail("cover.png", size=96)
            rendered.unlink()
            secret = Path(tmp) / "secret.webp"
            secret.write_bytes(b"PRIVATE DATA")
            try:
                rendered.symlink_to(secret)
            except OSError:
                self.skipTest("Symlink privilege unavailable")
            replacement = cache.thumbnail("cover.png", size=96)
            self.assertEqual(replacement, rendered)
            self.assertFalse(replacement.is_symlink())
            self.assertNotEqual(replacement.read_bytes(), b"PRIVATE DATA")
