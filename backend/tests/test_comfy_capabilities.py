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
