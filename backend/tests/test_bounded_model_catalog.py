"""Bounded model catalog scan used by advisory generation planning."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nexus_backend.config import NexusSettings
from nexus_backend.scanner import (
    MAX_BOUNDED_MODEL_DEPTH,
    scan_models_bounded,
)


class BoundedModelCatalogTests(unittest.TestCase):
    def settings(self, root: Path) -> NexusSettings:
        return NexusSettings(
            models_dir=root / "models",
            model_sources={"checkpoints": [root / "external-checkpoints"]},
            reference_model_sources=[],
        )

    def test_indexes_managed_and_configured_category_roots_without_opening_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            managed = root / "models" / "checkpoints" / "sdxl"
            external = root / "external-checkpoints" / "nested"
            managed.mkdir(parents=True)
            external.mkdir(parents=True)
            (managed / "managed.safetensors").write_bytes(b"")
            (external / "external.gguf").write_bytes(b"")
            result = scan_models_bounded(self.settings(root))
            self.assertTrue(result.complete)
            self.assertFalse(result.truncated)
            self.assertEqual(result.files_indexed, 2)
            self.assertEqual(result.catalog.categories["checkpoints"][0].name, "external.gguf")
            self.assertEqual(result.catalog.categories["checkpoints"][1].name, "managed.safetensors")
            self.assertEqual(result.catalog.categories["checkpoints"][0].source, "reference")
            self.assertEqual(result.catalog.categories["checkpoints"][1].source, "managed")
            self.assertTrue(all(not item.preview for item in result.catalog.categories["checkpoints"]))

    def test_skips_symlink_directory_and_file_and_marks_scan_incomplete(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            model_root = root / "models" / "checkpoints"
            model_root.mkdir(parents=True)
            (root / "external-checkpoints").mkdir()
            outside_path = Path(outside)
            (outside_path / "escaped.safetensors").write_bytes(b"")
            try:
                (model_root / "escape-dir").symlink_to(outside_path, target_is_directory=True)
                (model_root / "escape-file.safetensors").symlink_to(outside_path / "escaped.safetensors")
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not permitted in this environment")
            result = scan_models_bounded(self.settings(root))
            self.assertFalse(result.complete)
            self.assertFalse(result.truncated)
            self.assertEqual(result.files_indexed, 0)

    def test_depth_limit_marks_result_partial_without_following_deep_tree(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            deep = root / "models" / "checkpoints"
            for index in range(MAX_BOUNDED_MODEL_DEPTH + 2):
                deep = deep / f"level-{index}"
            deep.mkdir(parents=True)
            (deep / "deep.safetensors").write_bytes(b"")
            (root / "external-checkpoints").mkdir()
            result = scan_models_bounded(self.settings(root))
            self.assertFalse(result.complete)
            self.assertFalse(result.truncated)
            self.assertEqual(result.files_indexed, 0)

    def test_missing_configured_root_is_unverified_not_complete(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models" / "checkpoints").mkdir(parents=True)
            result = scan_models_bounded(self.settings(root))
            self.assertFalse(result.complete)
            self.assertEqual(result.files_indexed, 0)

    def test_entry_budget_returns_partial_catalog(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_root = root / "models" / "checkpoints"
            model_root.mkdir(parents=True)
            for index in range(6):
                (model_root / f"model-{index}.safetensors").write_bytes(b"")
            (root / "external-checkpoints").mkdir()
            with patch("nexus_backend.scanner.MAX_BOUNDED_MODEL_ENTRIES", 3):
                result = scan_models_bounded(self.settings(root))
            self.assertFalse(result.complete)
            self.assertTrue(result.truncated)
            self.assertLessEqual(result.entries_scanned, 4)
