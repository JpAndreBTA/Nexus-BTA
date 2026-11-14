"""Contract tests for the resource-bounded gallery selection."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from nexus_backend.gallery_listing import recent_media_files


class GalleryListingTests(unittest.TestCase):
    def test_most_recent_media_are_selected_without_other_extensions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "video").mkdir()
            for index in range(45):
                file = root / ("video" if index % 2 else "") / f"{index}.png"
                file.write_bytes(b"test")
                os.utime(file, (10000 + index, 10000 + index))
            (root / "ignored.txt").write_text("not a gallery file")
            result = recent_media_files(root, {".png", ".jpg"}, limit=5)
            self.assertEqual(len(result), 5)
            self.assertEqual([p.name for p, _ in result], ["44.png", "43.png", "42.png", "41.png", "40.png"])

    def test_video_sidecars_do_not_count_toward_gallery_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video = root / "video"
            video.mkdir()
            sidecar = video / "temporary_preview.png"
            sidecar.write_bytes(b"x")
            os.utime(sidecar, (30000, 30000))
            retained = root / "final.png"
            retained.write_bytes(b"y")
            os.utime(retained, (20000, 20000))
            result = recent_media_files(root, {".png"}, limit=1, skip_video_images=True)
            self.assertEqual([path.name for path, _ in result], ["final.png"])

    def test_empty_missing_and_disabled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(recent_media_files(root / "missing", {".png"}), [])
            self.assertEqual(recent_media_files(root, {".png"}, 0), [])


if __name__ == "__main__":
    unittest.main()
