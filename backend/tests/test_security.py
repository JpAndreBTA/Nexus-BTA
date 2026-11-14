"""Regression tests for local-first network defaults (no GPU or ComfyUI required)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from nexus_backend.security import cors_allowed_origins, require_explicit_network_ack, validate_remote_access_token
from nexus_backend.remote_gate import RemoteReadOnlyGate, allowed_remote_route


class SecurityBoundaryTests(unittest.TestCase):
    def test_default_cors_is_not_wildcard(self):
        with patch.dict(os.environ, {}, clear=True):
            origins = cors_allowed_origins()
        self.assertIn("http://127.0.0.1:3000", origins)
        self.assertNotIn("*", origins)
        self.assertNotIn("null", origins)

    def test_configured_cors_is_explicit(self):
        with patch.dict(os.environ, {
            "NEXUS_CORS_ORIGINS": "https://trusted.example,*,null,https://evil.example/path,http://user:pass@evil.example",
        }, clear=True):
            origins = cors_allowed_origins()
        self.assertIn("https://trusted.example", origins)
        self.assertNotIn("*", origins)
        self.assertNotIn("null", origins)
        self.assertNotIn("https://evil.example/path", origins)
        self.assertNotIn("http://user:pass@evil.example", origins)

    def test_loopback_can_start_without_ack(self):
        with patch.dict(os.environ, {}, clear=True):
            for host in ("127.0.0.1", "localhost", "::1"):
                require_explicit_network_ack(host)

    def test_network_interface_requires_explicit_ack(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                require_explicit_network_ack("0.0.0.0")

    def test_network_ack_does_not_mean_authentication(self):
        with patch.dict(os.environ, {"NEXUS_UNSAFE_NETWORK_ACK": "I_UNDERSTAND_THE_RISK"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "NEXUS_REMOTE_API_TOKEN"):
                require_explicit_network_ack("0.0.0.0")

    def test_nonloopback_requires_operator_token_and_ack(self):
        good_token = "Xf27E_48hY0P9sR5zQ7u2B1m6W4x3V8k9Y5"
        self.assertTrue(validate_remote_access_token(good_token))
        self.assertFalse(validate_remote_access_token("password123"))
        self.assertFalse(validate_remote_access_token("A" * 64))
        with patch.dict(os.environ, {"NEXUS_REMOTE_API_TOKEN": good_token}, clear=True):
            with self.assertRaises(RuntimeError):
                require_explicit_network_ack("0.0.0.0")
        with patch.dict(os.environ, {
            "NEXUS_REMOTE_API_TOKEN": good_token,
            "NEXUS_UNSAFE_NETWORK_ACK": "I_UNDERSTAND_THE_RISK",
        }, clear=True):
            require_explicit_network_ack("0.0.0.0")

    def test_asgi_actual_peer_prevents_direct_uvicorn_bypass(self):
        from nexus_backend.remote_gate import _socket_is_loopback
        self.assertFalse(_socket_is_loopback({
            "server": ("192.168.1.22", 7861), "client": ("192.168.1.50", 49152)
        }))
        self.assertFalse(_socket_is_loopback({
            "server": ("127.0.0.1", 7861), "client": ("203.0.113.10", 49152)
        }))
        self.assertFalse(_socket_is_loopback({
            "server": ("::ffff:192.168.1.22", 7861), "client": ("::1", 49152)
        }))
        self.assertTrue(_socket_is_loopback({
            "server": ("127.0.0.1", 7861), "client": ("127.0.0.1", 49152)
        }))

    def test_readonly_remote_allowlist(self):
        self.assertTrue(allowed_remote_route("GET", "/api/gallery/page"))
        self.assertTrue(allowed_remote_route("GET", "/api/gallery/thumb/x.png"))
        self.assertTrue(allowed_remote_route("GET", "/api/comfy/capabilities"))
        for path in ("/api/config", "/api/comfy/start", "/api/workflows",
                     "/api/workflows/foo/install-dependencies", "/outputs/a.png",
                     "/inputs/x.png", "/model-assets/item.bin", "/ui", "/app",
                     "/api/gallery/thumb/", "/api/gallery/video-thumb/"):
            self.assertFalse(allowed_remote_route("GET", path))
        self.assertFalse(allowed_remote_route("POST", "/api/gallery/page"))

    def test_remote_api_requires_bearer_and_denies_mutations_and_static_files(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from starlette.responses import JSONResponse

        token = "Xf27E_48hY0P9sR5zQ7u2B1m6W4x3V8k9Y5"
        app = FastAPI()
        app.add_middleware(RemoteReadOnlyGate)
        @app.get("/api/gallery/page")
        def gallery_page():
            return {"items": []}
        @app.post("/api/comfy/start")
        def run_comfy():
            raise AssertionError("unsafe action reached backend")
        @app.get("/outputs/private.png")
        def output():
            raise AssertionError("private asset reached backend")
        @app.get("/api/config")
        def config():
            raise AssertionError("private config reached backend")
        with patch.dict(os.environ, {
            "NEXUS_BACKEND_HOST": "0.0.0.0",
            "NEXUS_REMOTE_API_TOKEN": token,
        }, clear=True):
            with TestClient(app) as client:
                self.assertEqual(client.get("/api/gallery/page").status_code, 401)
                headers = {"Authorization": "Bearer " + token}
                self.assertEqual(client.get("/api/gallery/page", headers=headers).status_code, 200)
                self.assertEqual(client.post("/api/comfy/start", headers=headers).status_code, 403)
                self.assertEqual(client.get("/outputs/private.png", headers=headers).status_code, 403)
                self.assertEqual(client.get("/api/config", headers=headers).status_code, 403)
                self.assertEqual(client.get("/api/gallery/page", headers={"Authorization": "Bearer wrong"}).status_code, 401)
        with patch.dict(os.environ, {"NEXUS_BACKEND_HOST": "127.0.0.1"}, clear=True):
            self.assertEqual(TestClient(app).get("/api/gallery/page").status_code, 200)



if __name__ == "__main__":
    unittest.main()
