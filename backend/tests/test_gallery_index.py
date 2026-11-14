"""SQLite gallery indexing and page consistency regression suite."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from nexus_backend.gallery_index import GalleryIndex, default_gallery_db_path


class GalleryIndexTests(unittest.TestCase):
    def test_paginates_more_than_200_files_and_filters_video_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            db = Path(tmp) / "gallery.sqlite3"
            (root / "video").mkdir()
            (root / "video" / "preview.png").write_bytes(b"sidecar")
            for i in range(245):
                p = root / f"image-{i:03}.png"
                p.write_bytes(b"fake")
                os.utime(p, ns=((1000 + i) * 1_000_000_000, (1000 + i) * 1_000_000_000))
            index = GalleryIndex(root, db)
            first = index.page(limit=80)
            self.assertEqual(first.total, 245)
            self.assertEqual(len(first.records), 80)
            self.assertEqual(first.records[0].relative_path, "image-244.png")
            all_paths = list(first.records)
            cursor = first.next_cursor
            while cursor:
                batch = index.page(limit=80, cursor=cursor)
                all_paths.extend(batch.records)
                cursor = batch.next_cursor
            self.assertEqual(len(all_paths), 245)
            self.assertEqual(len({r.relative_path for r in all_paths}), 245)
            self.assertFalse(db.is_relative_to(root))

    def test_refreshes_changes_and_drops_deleted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            db = Path(tmp) / "gallery.sqlite3"
            old = root / "a.glb"
            old.write_bytes(b"model")
            index = GalleryIndex(root, db, refresh_seconds=100)
            result = index.page()
            self.assertEqual(result.records[0].media_type, "3d")
            old.unlink()
            (root / "b.mp3").write_bytes(b"music")
            self.assertEqual(index.page().total, 1)  # TTL cache still has old row
            refreshed = index.page(force_refresh=True)
            self.assertEqual(refreshed.total, 1)
            self.assertEqual(refreshed.records[0].relative_path, "b.mp3")
            self.assertEqual(refreshed.records[0].media_type, "audio")

    def test_broken_cursors_and_exposed_index_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            with self.assertRaises(ValueError):
                GalleryIndex(root, root / "bad.sqlite3")
            index = GalleryIndex(root, Path(tmp) / "safe.sqlite3")
            for cursor in ("unsafe/../../", "A" * 513, "e30", "***"):
                with self.assertRaises(ValueError):
                    index.page(cursor=cursor)
            for limit in (0, 101):
                with self.assertRaises(ValueError):
                    index.page(limit=limit)

    @unittest.skipUnless(hasattr(os, "symlink"), "Symlinks unavailable")
    def test_symlink_escape_is_not_indexed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            outside = Path(tmp) / "private.png"
            outside.write_bytes(b"secret")
            try:
                (root / "escaped.png").symlink_to(outside)
            except OSError:
                self.skipTest("Symlink privileges unavailable")
            index = GalleryIndex(root, Path(tmp) / "private-gallery.sqlite3")
            self.assertEqual(index.page().total, 0)

    def test_cache_path_uses_app_cache_not_public_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "outputs"
            root.mkdir()
            self.assertFalse(default_gallery_db_path(root).is_relative_to(root))
