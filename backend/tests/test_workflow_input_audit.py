"""Static Comfy API graph input inspection: bounded and no GPU dependencies."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from nexus_backend.workflow_input_audit import (
    MAX_WORKFLOW_BYTES, inspect_workflow_inputs, read_workflow_bounded,
)


REGISTRY = {
    "CheckpointLoaderSimple": {
        "input": {"required": {"ckpt_name": [["valid.safetensors", "other.safetensors"]]}},
        "output": ["MODEL", "CLIP", "VAE"],
    },
    "KSampler": {
        "input": {"required": {"model": ["MODEL"], "seed": ["INT"], "steps": ["INT"]},
                  "optional": {"sampler_name": [["euler", "dpmpp_2m"]]}},
        "output": ["LATENT"],
    },
}


class WorkflowInputAuditTests(unittest.TestCase):
    def test_valid_api_graph_has_static_only_validity(self):
        api = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "valid.safetensors"}},
            "2": {"class_type": "KSampler", "inputs": {
                "model": ["1", 0], "seed": 10, "steps": 20, "sampler_name": "euler"}},
        }
        result = inspect_workflow_inputs("qa", api, REGISTRY)
        self.assertEqual(result["status"], "static_valid")
        self.assertTrue(result["static_valid"])
        self.assertFalse(result["ready"], "static validity never certifies runnable workflow")
        self.assertEqual(result["node_count"], 2)
        self.assertEqual(result["findings"], [])

    def test_definite_broken_links_and_missing_class(self):
        api = {
            "1": {"class_type": "Nonexistent", "inputs": {}},
            "2": {"class_type": "KSampler", "inputs": {
                "model": ["missing-node", 0], "seed": 2, "steps": 20}},
        }
        result = inspect_workflow_inputs("qa", api, REGISTRY)
        self.assertEqual(result["status"], "issues")
        self.assertEqual(result["missing_classes"], 1)
        self.assertEqual(result["broken_links"], 1)

    def test_missing_inputs_and_choice_are_review_not_hard_failure(self):
        api = {
            "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "not-installed.safetensors"}},
            "2": {"class_type": "KSampler", "inputs": {"model": ["1", 0], "seed": 11}},
        }
        result = inspect_workflow_inputs("qa", api, REGISTRY)
        self.assertEqual(result["status"], "review")
        self.assertEqual(result["missing_required_inputs"], 1)
        self.assertEqual(result["choice_mismatches"], 1)
        self.assertEqual(result["broken_links"], 0)
        self.assertFalse(result["ready"])

    def test_offline_ui_and_node_limits_do_not_generate_false_positives(self):
        api = {"1": {"class_type": "CustomNode", "inputs": {"value": 42}}}
        offline = inspect_workflow_inputs("qa", api, None)
        self.assertEqual(offline["status"], "offline")
        self.assertEqual(offline["missing_classes"], 0)
        self.assertFalse(offline["runtime_checked"])
        ui = inspect_workflow_inputs("qa", {"nodes": [{"type": "CustomNode"}]}, REGISTRY)
        self.assertEqual(ui["status"], "ui_format")
        huge = {str(n): {"class_type": "CustomNode", "inputs": {}} for n in range(600)}
        self.assertEqual(inspect_workflow_inputs("qa", huge, REGISTRY)["status"], "unsupported")

    def test_bounded_file_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "workflow.json"
            path.write_text(json.dumps({"1": {"class_type": "KSampler", "inputs": {}}}), encoding="utf8")
            self.assertEqual(read_workflow_bounded(path)["1"]["class_type"], "KSampler")
            path.write_bytes(b" " * (MAX_WORKFLOW_BYTES + 1))
            with self.assertRaisesRegex(ValueError, "size limit"):
                read_workflow_bounded(path)

    def test_model_choices_and_registry_signature_report_only_menu_visibility(self):
        source = {
            "1": {
                "class_type": "CheckpointLoaderSimple",
                "inputs": {"ckpt_name": "not-advertised.safetensors"},
            },
            "2": {
                "class_type": "KSampler",
                "inputs": {"model": ["1", 0], "seed": 1, "steps": 25, "sampler_name": "euler"},
            },
        }
        registry = {
            "CheckpointLoaderSimple": {
                "input": {"required": {"ckpt_name": [["available.safetensors"]]}},
                "output": ["MODEL"],
            },
            "KSampler": {
                "input": {
                    "required": {"model": ["MODEL"], "seed": ["INT"], "steps": ["INT"]},
                    "optional": {"sampler_name": [["euler", "heun"]]},
                },
            },
        }
        first = inspect_workflow_inputs("qa", source, registry)
        self.assertEqual(first["model_choices_checked"], 1)
        self.assertEqual(first["model_choices_unavailable"], 1)
        self.assertEqual(first["choice_mismatches"], 1)
        self.assertEqual(first["findings"][0]["kind"], "model_not_advertised")
        self.assertEqual(len(first["schema_signature"]), 20)
        self.assertNotIn("not-advertised.safetensors", str(first))
        same = inspect_workflow_inputs("qa", source, registry)
        self.assertEqual(first["schema_signature"], same["schema_signature"])
        altered_registry = json.loads(json.dumps(registry))
        altered_registry["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0].append("new.safetensors")
        changed = inspect_workflow_inputs("qa", source, altered_registry)
        self.assertNotEqual(first["schema_signature"], changed["schema_signature"])
        offline = inspect_workflow_inputs("qa", source, None)
        self.assertIsNone(offline["schema_signature"])
        self.assertEqual(offline["model_choices_unavailable"], 0)
