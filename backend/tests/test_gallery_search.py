"""Indexed Gallery query/filter/sort tests across pages and schema migration."""
from __future__ import annotations

import base64
import json
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path

from nexus_backend.gallery_index import GalleryIndex


class GalleryIndexedSearchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "outputs"
        self.root.mkdir()
        self.db = Path(self.temp.name) / "index.sqlite3"
        (self.root / "renders").mkdir()
        for i, (name, kind) in enumerate([
            ("Ocean sky.png", "image"), ("Forest_100%.PNG", "image"),
            ("Ocean loop.mp4", "video"), ("forest river.glb", "3d"),
            ("ocean drums.mp3", "audio"), ("cliff.png", "image"),
            ("ocean tree.png", "image"),
        ]):
            path = self.root / "renders" / name
            path.write_bytes(b"fixture")
            os.utime(path, ns=((i + 1) * 1_000_000_000, (i + 1) * 1_000_000_000))
        self.index = GalleryIndex(self.root, self.db, refresh_seconds=3600)

    def test_query_matches_all_indexed_outputs_and_handles_literal_wildcards(self):
        results = self.index.page(limit=2, query="Ocean")
        self.assertEqual(results.total, 4)
        self.assertEqual(len(results.records), 2)
        next_page = self.index.page(limit=2, query="Ocean", cursor=results.next_cursor)
        self.assertEqual(len(next_page.records), 2)
        self.assertIsNone(next_page.next_cursor)
        self.assertEqual(self.index.page(query="100%").total, 1)
        self.assertEqual(self.index.page(query="Forest_").total, 1)

    def test_server_side_media_type_and_sort_keyset(self):
        self.assertEqual(self.index.page(media_type="video").total, 1)
        self.assertEqual(self.index.page(media_type="3d").total, 1)
        self.assertEqual(self.index.page(media_type="image").total, 4)
        for sort in ("oldest", "name", "type", "newest"):
            all_records = []
            cursor = None
            for _ in range(8):
                page = self.index.page(limit=2, cursor=cursor, sort_order=sort)
                all_records.extend(page.records)
                if not page.next_cursor:
                    break
                cursor = page.next_cursor
            self.assertEqual(len(all_records), 7, sort)
            self.assertEqual(len({item.relative_path for item in all_records}), 7, sort)
            if sort == "name":
                self.assertEqual([item.filename.casefold() for item in all_records],
                                 sorted(item.filename.casefold() for item in all_records))

    def test_cursors_cannot_reuse_other_filter_and_bad_queries(self):
        one = self.index.page(limit=1, query="Ocean")
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.index.page(limit=1, query="Forest", cursor=one.next_cursor)
        with self.assertRaises(ValueError):
            self.index.page(query="x" * 121)
        with self.assertRaises(ValueError):
            self.index.page(media_type="private")
        with self.assertRaises(ValueError):
            self.index.page(sort_order="DROP TABLE")

    def test_old_sqlite_schema_migrates_without_discarding_gallery(self):
        with sqlite3.connect(self.db) as conn:
            conn.execute("""CREATE TABLE media (
                relative_path TEXT PRIMARY KEY, modified_ns INTEGER NOT NULL,
                size_bytes INTEGER NOT NULL, media_type TEXT NOT NULL
            )""")
            conn.execute(
                "INSERT INTO media VALUES (?, ?, ?, ?)",
                ("renders/cliff.png", 1, 1, "image")
            )
        results = self.index.page(sort_order="name")
        self.assertEqual(results.total, 7)
        self.assertTrue(all(item.filename for item in results.records))
