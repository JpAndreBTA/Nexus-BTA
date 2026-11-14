"""Preflight must clone workflows, avoid actions and never claim runtime success."""
from __future__ import annotations

from copy import deepcopy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pydantic import ValidationError

from nexus_backend.config import NexusSettings
from nexus_backend.schemas import GenerateRequest
from nexus_backend.workflows import WorkflowRegistry, prepare_workflow_graph
from nexus_backend.workflow_preflight import WorkflowPreflightRequest, inspect_patched_preview


class WorkflowPreflightTests(unittest.TestCase):
    def sample_graph(self):
        return {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "original.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"model": ["1", 0], "seed": 100, "steps": 5, "cfg": 3},
            },
        }

    def registry(self):
        return {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["chosen.safetensors"]]}},
                "output": ["MODEL"],
            },
            "KSampler": {
                "input": {"required": {
                    "model": ["MODEL"], "seed": ["INT"], "steps": ["INT"], "cfg": ["FLOAT"],
                }},
                "output": ["LATENT"],
            },
        }

    def test_dry_run_patches_copy_without_changing_source(self):
        source = self.sample_graph()
        original = deepcopy(source)
        payload = WorkflowPreflightRequest(
            workflow_id="test-workflow", preset="SD",
            model_name="chosen.safetensors", steps=17, cfg=4.5, seed=42,
        )
        audited_graphs = []
        def audit(graph):
            audited_graphs.append(graph)
            return {"status": "complete", "found": 1}
        report = inspect_patched_preview(payload, source, self.registry(), audit)
        self.assertEqual(source, original, "read-only preflight must not mutate saved workflow")
        self.assertTrue(report["preview"])
        self.assertFalse(report["ready"])
        self.assertTrue(report["runtime_checked"])
        self.assertGreaterEqual(report["changed_nodes"], 1)
        self.assertGreaterEqual(report["changed_input_fields"], 1)
        self.assertNotIn("chosen.safetensors", str(report))
        self.assertNotIn("original.safetensors", str(report))
        self.assertEqual(report["workflow_id"], "test-workflow")
        self.assertEqual(report["model_sources"]["found"], 1)
        self.assertEqual(len(audited_graphs), 1)
        self.assertNotEqual(audited_graphs[0], source)

    def test_live_registry_and_preflight_share_the_same_graph_builder(self):
        source = self.sample_graph()
        original = deepcopy(source)
        request = GenerateRequest(
            workflow_id="qa", preset="SD", prompt="same prompt",
            model_name="chosen.safetensors", seed=42, steps=17, cfg=4.5,
        )
        assets = {"primary_model": "chosen.safetensors"}
        expected = prepare_workflow_graph(
            source, request.model_copy(deep=True), object_info=self.registry(), assets=assets,
        )
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workflow.json"
            path.write_text(json.dumps(source), encoding="utf8")
            settings = NexusSettings(
                project_root=Path(folder),
                workflows_dir=Path(folder) / "workflows",
            )
            actual = WorkflowRegistry(settings).load_api_workflow(
                path, request.model_copy(deep=True), self.registry(), assets=assets,
            )
        self.assertEqual(actual, expected)
        self.assertEqual(source, original)

    def test_offline_never_claims_compatibility(self):
        report = inspect_patched_preview(WorkflowPreflightRequest(workflow_id="qa"), self.sample_graph(), None)
        self.assertEqual(report["status"], "offline")
        self.assertFalse(report["ready"])
        self.assertEqual(report["missing_classes"], 0)

    def test_dry_run_rejects_unsupported_and_oversized_payloads(self):
        with self.assertRaises(ValidationError):
            WorkflowPreflightRequest(workflow_id="bad/name")
        with self.assertRaises(ValidationError):
            WorkflowPreflightRequest(workflow_id="qa", prompt="x" * 5000)
        with self.assertRaises(ValidationError):
            WorkflowPreflightRequest(workflow_id="qa", uploaded_mask="data:image/png;base64,AAA")
        with self.assertRaises(ValidationError):
            WorkflowPreflightRequest(workflow_id="qa", cfg=float("nan"))
        invalid = inspect_patched_preview(
            WorkflowPreflightRequest(workflow_id="qa"),
            {"nodes": [{"type": "KSampler"}]}, self.registry(),
        )
        self.assertEqual(invalid["status"], "unsupported")
        with self.assertRaisesRegex(ValueError, "node budget"):
            inspect_patched_preview(
                WorkflowPreflightRequest(workflow_id="qa"),
                {str(i): {"class_type": "KSampler", "inputs": {}} for i in range(600)},
                self.registry(),
            )
