"""Regression tests for local-first network defaults (no GPU or ComfyUI required)."""
from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from nexus_backend.security import cors_allowed_origins, require_explicit_network_ack


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
            require_explicit_network_ack("0.0.0.0")


if __name__ == "__main__":
    unittest.main()
