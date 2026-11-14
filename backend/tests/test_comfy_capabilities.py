"""FastAPI contract: no Comfy process launch or dependency installation."""
from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from nexus_backend.comfy_capabilities import (
    make_comfy_capability_router, parse_requested_classes, summarize_capabilities,
)


class ComfyCapabilityTests(unittest.TestCase):
    def test_offline_is_not_misdiagnosed_as_missing(self):
        report = summarize_capabilities(["KSampler", "CLIPTextEncode"], None)
        self.assertEqual(report["status"], "offline")
        self.assertFalse(report["runtime_checked"])
        self.assertEqual(report["missing"], [])
        self.assertEqual(report["unverified"], ["KSampler", "CLIPTextEncode"])

    def test_registered_and_missing_are_distinct(self):
        report = summarize_capabilities(["KSampler", "SuperCustom"], {"KSampler": {"input": {}}, "VAEDecode": {}})
        self.assertEqual(report["status"], "missing")
        self.assertEqual(report["available"], ["KSampler"])
        self.assertEqual(report["missing"], ["SuperCustom"])
        self.assertEqual(report["registered_count"], 2)
        self.assertTrue(summarize_capabilities(["KSampler"], {"KSampler": {}})["ready"])

    def test_validation_deduplicates_and_rejects_overflow(self):
        self.assertEqual(parse_requested_classes("KSampler, VAEDecode,KSampler"), ["KSampler", "VAEDecode"])
        for invalid in ("", "bad/class", "KSampler,,SaveImage", ",".join(f"Class{i}" for i in range(70))):
            with self.assertRaises(ValueError):
                parse_requested_classes(invalid)

    def test_workflow_check_uses_actual_class_list_and_never_updates_runtime(self):
        requested_workflows = []
        calls = []
        def lookup(workflow_id: str):
            requested_workflows.append(workflow_id)
            return ["KSampler", "VAEDecode", "CustomMissing", "KSampler"] if workflow_id == "my-wf" else None

        async def registry_reader():
            calls.append("registry")
            return {"KSampler": {}, "VAEDecode": {}}

        app = FastAPI()
        app.include_router(make_comfy_capability_router(registry_reader, workflow_lookup=lookup))
        with TestClient(app) as client:
            response = client.get("/api/comfy/workflow-capabilities?workflow_id=my-wf")
            self.assertEqual(response.status_code, 200)
            report = response.json()
            self.assertEqual(report["requested"], ["KSampler", "VAEDecode", "CustomMissing"])
            self.assertEqual(report["missing"], ["CustomMissing"])
            self.assertFalse(report["ready"])
            self.assertEqual(report["required_count"], 3)
            self.assertEqual(calls, ["registry"])
            self.assertEqual(client.get("/api/comfy/workflow-capabilities?workflow_id=../bad").status_code, 400)
            self.assertEqual(client.get("/api/comfy/workflow-capabilities?workflow_id=none").status_code, 404)
            self.assertEqual(calls, ["registry"], "invalid and missing workflow must not query runtime")
            self.assertEqual(requested_workflows, ["my-wf", "none"])

    def test_workflow_unverified_offline_and_oversized_are_distinct(self):
        async def offline_reader():
            return None
        app = FastAPI()
        app.include_router(make_comfy_capability_router(
            offline_reader, workflow_lookup=lambda wid: ["CustomNode"] if wid == "small" else [f"Node{i}" for i in range(300)]
        ))
        with TestClient(app) as client:
            offline = client.get("/api/comfy/workflow-capabilities?workflow_id=small").json()
            self.assertEqual(offline["status"], "offline")
            self.assertEqual(offline["missing"], [])
            oversized = client.get("/api/comfy/workflow-capabilities?workflow_id=big").json()
            self.assertEqual(oversized["status"], "unverified")
            self.assertEqual(oversized["required_count"], 300)
            self.assertFalse(oversized["ready"])
            self.assertLessEqual(len(oversized["requested"]), 16)


    def test_workflow_input_route_bounds_files_and_flags_choice_review(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "workflow.json"
            path.write_text(json.dumps({
                "1": {"class_type": "CheckpointLoaderSimple", "inputs": {
                    "ckpt_name": "not-present.safetensors"}},
                "2": {"class_type": "KSampler", "inputs": {
                    "model": ["1", 0], "seed": 4}},
            }), encoding="utf8")
            calls = []
            def lookup(key):
                return path if key == "qa" else None
            async def registry():
                calls.append("registry")
                return {
                    "CheckpointLoaderSimple": {
                        "input": {"required": {"ckpt_name": [["present.safetensors"]]}},
                        "output": ["MODEL"],
                    },
                    "KSampler": {"input": {"required": {
                        "model": ["MODEL"], "seed": ["INT"], "steps": ["INT"],
                    }}},
                }
            app = FastAPI()
            app.include_router(make_comfy_capability_router(
                registry, workflow_path_lookup=lookup,
            ))
            client = TestClient(app)
            report = client.get("/api/comfy/workflow-inputs?workflow_id=qa").json()
            self.assertEqual(report["status"], "review")
            self.assertEqual(report["missing_required_inputs"], 1)
            self.assertEqual(report["choice_mismatches"], 1)
            self.assertFalse(report["ready"])
            self.assertEqual(calls, ["registry"])
            self.assertEqual(client.get("/api/comfy/workflow-inputs?workflow_id=unknown").status_code, 404)
            self.assertEqual(client.get("/api/comfy/workflow-inputs?workflow_id=../invalid").status_code, 400)
            self.assertEqual(calls, ["registry"])
            path.write_bytes(b" " * (4 * 1024 * 1024 + 1))
            self.assertEqual(client.get("/api/comfy/workflow-inputs?workflow_id=qa").status_code, 422)
            self.assertEqual(calls, ["registry"])

    def test_workflow_input_route_offline_stays_unverified(self):
        with tempfile.TemporaryDirectory() as folder:
            file = Path(folder) / "workflow.json"
            file.write_text('{"1":{"class_type":"Unavailable","inputs":{}}}', encoding="utf8")
            async def offline():
                return None
            app = FastAPI()
            app.include_router(make_comfy_capability_router(
                offline, workflow_path_lookup=lambda _: file,
            ))
            report = TestClient(app).get("/api/comfy/workflow-inputs?workflow_id=qa").json()
            self.assertEqual(report["status"], "offline")
            self.assertEqual(report["missing_classes"], 0)
            self.assertFalse(report["runtime_checked"])
            self.assertFalse(report["ready"])

    def test_preflight_endpoint_does_not_start_or_submit_comfy(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "qa.json"
            path.write_text(json.dumps({
                "1": {
                    "class_type": "CheckpointLoaderSimple",
                    "inputs": {"ckpt_name": "old.safetensors"},
                },
                "2": {
                    "class_type": "KSampler",
                    "inputs": {"model": ["1", 0], "seed": 1, "steps": 4, "cfg": 4},
                },
            }), encoding="utf8")
            calls = []
            async def read_registry():
                calls.append("object_info_only")
                return {
                    "CheckpointLoaderSimple": {
                        "input": {"required": {"ckpt_name": [["selected.safetensors"]]}},
                        "output": ["MODEL"],
                    },
                    "KSampler": {"input": {"required": {
                        "model": ["MODEL"], "seed": ["INT"], "steps": ["INT"], "cfg": ["FLOAT"],
                    }}},
                }
            app = FastAPI()
            app.include_router(make_comfy_capability_router(
                read_registry,
                workflow_path_lookup=lambda name: path if name == "qa" else None,
            ))
            original = path.read_bytes()
            with TestClient(app) as client:
                response = client.post("/api/comfy/workflow-preflight", json={
                    "workflow_id": "qa", "preset": "SD", "model_name": "selected.safetensors",
                    "steps": 22, "seed": 42,
                })
                self.assertEqual(response.status_code, 200, response.text)
                preview = response.json()
                self.assertTrue(preview["preview"])
                self.assertFalse(preview["ready"])
                self.assertGreaterEqual(preview["changed_input_fields"], 1)
                self.assertNotIn("selected.safetensors", response.text)
                self.assertNotIn("old.safetensors", response.text)
                self.assertEqual(calls, ["object_info_only"])
                self.assertEqual(path.read_bytes(), original)
                self.assertEqual(client.post("/api/comfy/workflow-preflight", json={
                    "workflow_id": "missing",
                }).status_code, 404)
                self.assertEqual(client.post("/api/comfy/workflow-preflight", json={
                    "workflow_id": "../unsafe",
                }).status_code, 422)
                self.assertEqual(calls, ["object_info_only"])

    def test_readonly_http_backend_and_offline_behavior(self):
        calls = []
        async def registry_reader():
            calls.append("read")
            return {"KSampler": {}, "VAEDecode": {}}

        app = FastAPI()
        app.include_router(make_comfy_capability_router(registry_reader))
        with TestClient(app) as client:
            response = client.get("/api/comfy/capabilities?classes=KSampler,Missing,KSampler")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["missing"], ["Missing"])
            self.assertEqual(calls, ["read"])
            self.assertEqual(client.get("/api/comfy/capabilities?classes=KSampler,,X").status_code, 400)
            self.assertEqual(calls, ["read"], "invalid input must not query Comfy")

        async def offline_reader():
            return None
        app2 = FastAPI()
        app2.include_router(make_comfy_capability_router(offline_reader))
        self.assertEqual(TestClient(app2).get("/api/comfy/capabilities?classes=KSampler").json()["status"], "offline")
