"""No GPU/network required: video poster decoder and gallery HTTP contract."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus_backend.gallery_video import GalleryVideoPosterCache, frame_command
from nexus_backend.gallery_api import make_gallery_router


class VideoPosterTests(unittest.TestCase):
    def test_decoder_arguments_are_shell_free_and_restricted(self):
        args = frame_command("ffmpeg", Path("local.mp4"), Path("out.png"), 320)
        self.assertEqual(args[0], "ffmpeg")
        self.assertEqual(args[args.index("-protocol_whitelist") + 1], "file,pipe")
        self.assertEqual(args[args.index("-frames:v") + 1], "1")
        self.assertNotIn("-i http://", " ".join(args))
        self.assertEqual(args[-1], "out.png")
        with self.assertRaises(ValueError):
            frame_command("ffmpeg", Path("local.mp4"), Path("out.png"), 4096)

    def test_poster_cache_success_invalid_media_and_failed_decoder(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            (root / "scene.mp4").write_bytes(b"mock video")
            invocations = []
            def fake_run(args):
                invocations.append(args)
                Image.new("RGB", (320, 180), (30, 80, 140)).save(args[-1])
            store = GalleryVideoPosterCache(
                root, cache_root=Path(tmp) / "cache",
                executable="ffmpeg-test", runner=fake_run,
            )
            result = store.poster("scene.mp4")
            self.assertEqual(result.suffix, ".webp")
            self.assertFalse(result.is_relative_to(root))
            with Image.open(result) as poster:
                self.assertLessEqual(max(poster.size), 320)
            self.assertEqual(store.poster("scene.mp4"), result)
            self.assertEqual(len(invocations), 1, "cached poster must not decode twice")
            with self.assertRaises(ValueError):
                store.poster("../secret.mp4")
            with self.assertRaises(ValueError):
                store.poster("scene.mp4", size=1024)
            with self.assertRaises(ValueError):
                store.poster("bad.png")
            broken = GalleryVideoPosterCache(root, cache_root=Path(tmp) / "broken", executable="mock",
                                            runner=lambda _: (_ for _ in ()).throw(OSError("Decoder failed")))
            with self.assertRaises(ValueError):
                broken.poster("scene.mp4")
            self.assertFalse(list(broken.cache_root.glob("*.webp")))

    def test_http_route_with_mock_decoder_and_missing_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            (root / "scene.mp4").write_bytes(b"mock")
            def fake_run(args):
                Image.new("RGB", (320, 180), "navy").save(args[-1])
            poster_cache = GalleryVideoPosterCache(root, cache_root=Path(tmp) / "posters",
                                                    executable="ffmpeg-test", runner=fake_run)
            app = FastAPI()
            app.include_router(make_gallery_router(lambda: root, lambda _: {}, video_cache_factory=lambda _: poster_cache))
            client = TestClient(app)
            gallery = client.get("/api/gallery/page").json()
            self.assertEqual(gallery["total"], 1)
            self.assertIn("/api/gallery/video-thumb/", gallery["items"][0]["thumb"])
            ok = client.get(gallery["items"][0]["thumb"])
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(ok.headers["content-type"], "image/webp")
            self.assertEqual(client.get("/api/gallery/video-thumb/scene.mp4?size=800").status_code, 422)
            self.assertEqual(client.get("/api/gallery/video-thumb/%2E%2E/private.mp4").status_code, 404)
