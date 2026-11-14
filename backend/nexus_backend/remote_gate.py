"""Fail-closed ASGI authorization for opt-in, read-only non-loopback API.

No cookies, shared remote UI, or remote process/install/file editing access.
Token never appears in URLs, query strings, logs or server response bodies.
This is an extra layer, NOT a replacement for an authenticated TLS gateway.
"""
from __future__ import annotations

import hmac
import ipaddress
import os
from typing import Any

from .security import is_loopback_listener, validate_remote_access_token

READONLY_PATHS = frozenset({
    "/api/gallery/page",
    "/api/comfy/capabilities",
})
READONLY_PREFIXES = ("/api/gallery/thumb/", "/api/gallery/video-thumb/")


def allowed_remote_route(method: str, path: str) -> bool:
    return (
        method in {"GET", "HEAD"}
        and (
            path in READONLY_PATHS
            or any(path.startswith(prefix) and len(path) > len(prefix) for prefix in READONLY_PREFIXES)
        )
    )


def _socket_is_loopback(scope: dict[str, Any]) -> bool:
    for key in ("server", "client"):
        endpoint = scope.get(key)
        if not endpoint or not isinstance(endpoint, (tuple, list)) or not endpoint[0]:
            continue
        hostname = str(endpoint[0])
        if is_loopback_listener(hostname):
            continue
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            # Starlette TestClient uses a non-IP synthetic host. In
            # production Uvicorn supplies real peer/socket IP addresses.
            continue
        if address.ipv4_mapped:
            address = address.ipv4_mapped
        if not address.is_loopback:
            return False
    return True


class RemoteReadOnlyGate:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        kind = scope.get("type")
        if kind not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return
        # The actual app listener is local by default. Avoid locking out /ui and
        # /app on the intended desktop deployment.
        host = os.environ.get("NEXUS_BACKEND_HOST", "127.0.0.1")
        # Also inspect real ASGI socket and peer addresses. A direct
        # uvicorn --host 0.0.0.0 launch may bypass run_backend.py and
        # never set NEXUS_BACKEND_HOST. HTTP Host/X-Forwarded-* are untrusted.
        if is_loopback_listener(host) and _socket_is_loopback(scope):
            await self.app(scope, receive, send)
            return
        if kind == "websocket":
            await send({"type": "websocket.close", "code": 4403})
            return

        method = scope.get("method", "").upper()
        path = scope.get("path", "")
        token = os.environ.get("NEXUS_REMOTE_API_TOKEN", "")
        authenticated = False
        if validate_remote_access_token(token):
            values = [
                value for key, value in scope.get("headers", [])
                if key.lower() == b"authorization"
            ]
            if len(values) == 1:
                try:
                    supplied = values[0].decode("ascii")
                    if supplied.startswith("Bearer "):
                        candidate = supplied[7:]
                        authenticated = hmac.compare_digest(candidate, token)
                except UnicodeDecodeError:
                    pass

        if not authenticated or not allowed_remote_route(method, path):
            # Deny all local files (/inputs, /models, /outputs), frontend assets
            # and every endpoint capable of mutation/restart/model installation.
            status = 401 if not authenticated else 403
            body = b'{"detail":"Remote operation not authorized."}'
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"cache-control", b"no-store"),
                    (b"x-content-type-options", b"nosniff"),
                ],
            })
            await send({"type": "http.response.body", "body": body})
            return
        await self.app(scope, receive, send)
