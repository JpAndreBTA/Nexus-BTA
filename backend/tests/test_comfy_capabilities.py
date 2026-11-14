"""FastAPI contract: no Comfy process launch or dependency installation."""
from __future__ import annotations

import unittest

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
