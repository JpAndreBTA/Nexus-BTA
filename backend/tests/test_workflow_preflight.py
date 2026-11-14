"""Preflight must clone workflows, avoid actions and never claim runtime success."""
from __future__ import annotations

from copy import deepcopy
import unittest
from unittest.mock import patch

from pydantic import ValidationError

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
        report = inspect_patched_preview(payload, source, self.registry())
        self.assertEqual(source, original, "read-only preflight must not mutate saved workflow")
        self.assertTrue(report["preview"])
        self.assertFalse(report["ready"])
        self.assertTrue(report["runtime_checked"])
        self.assertGreaterEqual(report["changed_nodes"], 1)
        self.assertGreaterEqual(report["changed_input_fields"], 1)
        self.assertNotIn("chosen.safetensors", str(report))
        self.assertNotIn("original.safetensors", str(report))
        self.assertEqual(report["workflow_id"], "test-workflow")

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
