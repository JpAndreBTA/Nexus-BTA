"""FastAPI contract test for paginated gallery (no ComfyUI or GPU)."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus_backend.gallery_api import make_gallery_router


class GalleryHTTPContractTests(unittest.TestCase):
    def test_multiple_pages_legacy_independent_and_no_absolute_path_leaks(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            for i in range(120):
                path = root / f"render {i:03}.png"
                path.write_bytes(b"mock")
                os.utime(path, ns=((2000 + i) * 1_000_000_000, (2000 + i) * 1_000_000_000))
            app = FastAPI()
            app.include_router(make_gallery_router(lambda: root, lambda _: {"model": "Test Model"}))
            client = TestClient(app)
            one = client.get("/api/gallery/page?limit=50")
            self.assertEqual(one.status_code, 200)
            first = one.json()
            self.assertEqual(first["total"], 120)
            self.assertEqual(len(first["items"]), 50)
            self.assertEqual(first["items"][0]["filename"], "render 119.png")
            self.assertEqual(first["items"][0]["image"], "/outputs/render%20119.png")
            self.assertEqual(first["items"][0]["path"], "render 119.png")
            self.assertNotIn(str(root), repr(first))
            two = client.get("/api/gallery/page", params={"limit": 50, "cursor": first["next_cursor"]})
            self.assertEqual(two.status_code, 200)
            self.assertEqual(len(two.json()["items"]), 50)
            three = client.get("/api/gallery/page", params={"limit": 50, "cursor": two.json()["next_cursor"]})
            self.assertEqual(three.status_code, 200)
            self.assertEqual(len(three.json()["items"]), 20)
            self.assertIsNone(three.json()["next_cursor"])
            self.assertEqual(len({
                item["relative_path"]
                for item in first["items"] + two.json()["items"] + three.json()["items"]
            }), 120)

    def test_bad_inputs_empty_root_and_update_refresh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            app = FastAPI()
            app.include_router(make_gallery_router(lambda: root, lambda _: {}))
            client = TestClient(app)
            self.assertEqual(client.get("/api/gallery/page").json()["total"], 0)
            self.assertEqual(client.get("/api/gallery/page?limit=101").status_code, 422)
            self.assertEqual(client.get("/api/gallery/page?cursor=unsafe!").status_code, 400)
            root.mkdir()
            (root / "changed.glb").write_bytes(b"3d")
            result = client.get("/api/gallery/page?refresh=true").json()
            self.assertEqual(result["total"], 1)
            self.assertEqual(result["items"][0]["media_type"], "3d")
