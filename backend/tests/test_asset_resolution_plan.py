"""Pure logical model selection from an already-built catalog."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nexus_backend.asset_resolver import (
    resolve_generation_assets,
    resolve_generation_assets_from_catalog,
)
from nexus_backend.config import NexusSettings
from nexus_backend.schemas import GenerateRequest, ModelCatalog, ModelFile


class AssetResolutionPlanTests(unittest.TestCase):
    def model(self):
        return ModelFile(
            name="model.safetensors",
            path="/virtual/models/checkpoints/model.safetensors",
            relative_path="model.safetensors",
            category="checkpoints",
            folder="Root",
            extension=".safetensors",
            size_bytes=123,
            modified="2026-11-14T00:00:00+00:00",
            tags=["checkpoints", "safetensors"],
            source="managed",
        )

    def test_pure_catalog_resolver_selects_model_without_filesystem_access_or_request_mutation(self):
        catalog = ModelCatalog(
            root="/virtual/models",
            categories={"checkpoints": [self.model()]},
            total_files=1,
        )
        request = GenerateRequest(preset="custom", model_name="model.safetensors")
        before = request.model_dump()
        with patch("nexus_backend.asset_resolver.scan_models", side_effect=AssertionError("pure helper must not scan")):
            resolved = resolve_generation_assets_from_catalog(catalog.categories, request)
        self.assertEqual(resolved, {"primary_model": "model.safetensors"})
        self.assertEqual(request.model_dump(), before)

    def test_filesystem_adapter_delegates_to_same_catalog_selection_rules(self):
        catalog = ModelCatalog(
            root="/virtual/models",
            categories={"checkpoints": [self.model()]},
            total_files=1,
        )
        request = GenerateRequest(preset="custom", model_name="model.safetensors")
        with tempfile.TemporaryDirectory() as folder:
            settings = NexusSettings(models_dir=Path(folder) / "models", model_sources={})
            with patch("nexus_backend.asset_resolver.scan_models", return_value=catalog) as scan:
                live_adapter = resolve_generation_assets(settings, request)
            planned = resolve_generation_assets_from_catalog(catalog.categories, request)
        self.assertEqual(live_adapter, planned)
        scan.assert_called_once_with(settings, include_references=False)

    def test_empty_catalog_does_not_invent_a_selected_file(self):
        request = GenerateRequest(preset="custom", model_name="")
        self.assertEqual(resolve_generation_assets_from_catalog({}, request), {})
