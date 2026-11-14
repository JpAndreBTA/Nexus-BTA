"""Logical asset planning never performs generation or hides partial catalogs."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from nexus_backend.config import NexusSettings
from nexus_backend.schemas import GenerateRequest
from nexus_backend.generation_planner import plan_generation_assets


class LogicalAssetPlanTests(unittest.TestCase):
    def settings(self, root: Path, *, external_exists: bool = True) -> NexusSettings:
        (root / "models" / "checkpoints").mkdir(parents=True, exist_ok=True)
        external = root / "external-checkpoints"
        if external_exists:
            external.mkdir(parents=True, exist_ok=True)
        return NexusSettings(
            models_dir=root / "models",
            model_sources={"checkpoints": [external]},
            reference_model_sources=[],
        )

    def test_complete_catalog_returns_logical_asset_selection_without_mutating_request(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models" / "checkpoints" / "model.safetensors").write_bytes(b"")
            settings = self.settings(root)
            request = GenerateRequest(preset="custom", model_name="model.safetensors")
            before = request.model_dump()
            plan = plan_generation_assets(settings, request)
            self.assertEqual(plan.status, "resolved")
            self.assertTrue(plan.catalog_complete)
            self.assertFalse(plan.truncated)
            self.assertTrue(plan.assets["primary_model"].endswith("model.safetensors"))
            self.assertEqual(request.model_dump(), before)
            self.assertIn("No prompt was submitted", plan.note)

    def test_partial_catalog_keeps_tentative_selection_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "models" / "checkpoints" / "model.safetensors").write_bytes(b"")
            settings = self.settings(root, external_exists=False)
            plan = plan_generation_assets(
                settings, GenerateRequest(preset="custom", model_name="model.safetensors"),
            )
            self.assertEqual(plan.status, "partial")
            self.assertFalse(plan.catalog_complete)
            self.assertTrue(plan.assets.get("primary_model"))

    def test_complete_empty_catalog_does_not_claim_a_model_was_resolved(self):
        with tempfile.TemporaryDirectory() as folder:
            settings = self.settings(Path(folder))
            plan = plan_generation_assets(settings, GenerateRequest(preset="custom"))
            self.assertEqual(plan.status, "no_assets")
            self.assertEqual(plan.assets, {})
            self.assertTrue(plan.catalog_complete)
