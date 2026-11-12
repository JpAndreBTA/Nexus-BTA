"""Network boundary helpers for the local-first Nexus API.

CORS is a browser cross-origin policy, NOT authentication. Keep API server
loopback-bound unless a separate authenticated gateway is installed.
"""
from __future__ import annotations

import os
from urllib.parse import urlsplit


def cors_allowed_origins() -> list[str]:
    """Explicit browser origins only. The local React dev proxy usually needs none."""
    defaults = ["http://127.0.0.1:3000", "http://localhost:3000"]
    configured = os.environ.get("NEXUS_CORS_ORIGINS", "")
    for candidate in configured.split(","):
        value = candidate.strip().rstrip("/")
        if not value or value in {"*", "null"}:
            continue
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            continue
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            continue
        if value not in defaults:
            defaults.append(value)
    return defaults


def require_explicit_network_ack(host: str) -> None:
    """Fail closed for non-loopback listeners until a genuine auth gateway exists."""
    if host.strip().lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}:
        return
    if os.environ.get("NEXUS_UNSAFE_NETWORK_ACK") != "I_UNDERSTAND_THE_RISK":
        raise RuntimeError(
            "Nexus does not provide public API authentication yet. Non-loopback "
            "listeners are disabled by default. Use localhost or explicitly set "
            "NEXUS_UNSAFE_NETWORK_ACK=I_UNDERSTAND_THE_RISK only behind a trusted "
            "network and an independently authenticated gateway."
        )
