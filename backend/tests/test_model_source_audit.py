"""Bounded model source audit: names only, configured roots, no symlink traversal."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nexus_backend.config import NexusSettings
from nexus_backend.model_source_audit import (
    MAX_SCAN_DEPTH, audit_model_references,
)


class ModelSourceAuditTests(unittest.TestCase):
    def make_settings(self, root: Path) -> NexusSettings:
        return NexusSettings(
            project_root=root,
            models_dir=root / "models",
            model_sources={"checkpoints": [root / "external-checkpoints"]},
            reference_model_sources=[],
        )

    def test_finds_only_exact_configured_relative_model_without_echoing_names_or_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            model_root = root / "models" / "checkpoints"
            model_root.joinpath("nested").mkdir(parents=True)
            (root / "models" / "vae").mkdir(parents=True)
            (model_root / "nested" / "model.safetensors").write_bytes(b"")
            report = audit_model_references({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": "nested/model.safetensors",
                }},
                "2": {"class_type": "VAELoader", "inputs": {
                    "vae_name": "missing-vae.safetensors",
                }},
            }, self.make_settings(root))
            self.assertEqual(report["status"], "issues")
            self.assertEqual(report["found"], 1)
            self.assertEqual(report["not_found"], 1)
            self.assertEqual(report["unverified"], 0)
            self.assertEqual(report["entries"][0]["status"], "found")
            self.assertEqual(report["entries"][1]["status"], "not_found")
            self.assertNotIn("model.safetensors", str(report))
            self.assertNotIn(str(root), str(report))
            self.assertEqual(report["roots_checked"], 1)

    def test_checks_additional_configured_model_sources(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            external = root / "external-checkpoints"
            external.mkdir()
            (external / "external.safetensors").write_bytes(b"")
            report = audit_model_references({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": "external.safetensors",
                }},
            }, self.make_settings(root))
            self.assertEqual(report["found"], 1)
            self.assertEqual(report["entries"][0]["status"], "found")
            self.assertIn("model_sources/checkpoints[1]", report["entries"][0]["sources"])

    def test_skips_symlink_escapes_and_does_not_claim_missing_when_scan_is_incomplete(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            model_root = root / "models" / "checkpoints"
            model_root.mkdir(parents=True)
            external = Path(outside)
            (external / "outside.safetensors").write_bytes(b"")
            link = model_root / "external-link"
            try:
                link.symlink_to(external, target_is_directory=True)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not permitted in this environment")
            report = audit_model_references({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": "outside.safetensors",
                }},
            }, self.make_settings(root))
            self.assertEqual(report["entries"][0]["status"], "unverified")
            self.assertEqual(report["not_found"], 0)
            self.assertNotIn(str(external), str(report))

    def test_skips_symlink_model_file(self):
        with tempfile.TemporaryDirectory() as folder, tempfile.TemporaryDirectory() as outside:
            root = Path(folder)
            model_root = root / "models" / "checkpoints"
            model_root.mkdir(parents=True)
            external_file = Path(outside) / "external.safetensors"
            external_file.write_bytes(b"")
            link = model_root / "external.safetensors"
            try:
                link.symlink_to(external_file)
            except (OSError, NotImplementedError):
                self.skipTest("symlink creation is not permitted in this environment")
            report = audit_model_references({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": "external.safetensors",
                }},
            }, self.make_settings(root))
            self.assertEqual(report["entries"][0]["status"], "unverified")
            self.assertEqual(report["not_found"], 0)

    def test_empty_graph_is_not_applicable_and_absolute_paths_are_not_opened(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            empty = audit_model_references({}, self.make_settings(root))
            self.assertEqual(empty["status"], "not_applicable")
            absolute = audit_model_references({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": str(root / "private.safetensors"),
                }},
            }, self.make_settings(root))
            self.assertEqual(absolute["references_checked"], 1)
            self.assertEqual(absolute["unverified"], 1)
            self.assertEqual(absolute["entries"][0]["reason"], "unsafe_or_unsupported_reference")
            self.assertNotIn(str(root), str(absolute))

    def test_depth_limit_is_explicit_and_conservative(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            deep = root / "models" / "checkpoints"
            for index in range(MAX_SCAN_DEPTH + 2):
                deep = deep / f"level-{index}"
            deep.mkdir(parents=True)
            (deep / "too-deep.safetensors").write_bytes(b"")
            report = audit_model_references({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": "too-deep.safetensors",
                }},
            }, self.make_settings(root))
            self.assertEqual(report["entries"][0]["status"], "unverified")
