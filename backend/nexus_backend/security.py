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


def is_loopback_listener(host: str) -> bool:
    return host.strip().lower() in {"127.0.0.1", "localhost", "::1", "[::1]"}


def validate_remote_access_token(value: str | None) -> bool:
    """Require an operator-provided, nontrivial token. No default credentials."""
    return bool(
        value and 32 <= len(value) <= 512 and value.isascii()
        and not value.isspace() and not any(ch.isspace() for ch in value)
        and len(set(value)) >= 12
    )


def require_explicit_network_ack(host: str) -> None:
    """Non-loopback API is experimental, read-only and requires operator creds.

    This does not enable safe raw public HTTP. Use a trusted HTTPS gateway.
    """
    if is_loopback_listener(host):
        return
    if os.environ.get("NEXUS_UNSAFE_NETWORK_ACK") != "I_UNDERSTAND_THE_RISK":
        raise RuntimeError(
            "Non-loopback Nexus listeners require an explicit risk acknowledgment "
            "and a trusted HTTPS gateway; public exposure remains unsupported."
        )
    if not validate_remote_access_token(os.environ.get("NEXUS_REMOTE_API_TOKEN")):
        raise RuntimeError(
            "Non-loopback Nexus listeners require NEXUS_REMOTE_API_TOKEN "
            "(a random, private token of at least 32 characters). "
            "All remote operations except the read-only allowlist are denied."
        )
