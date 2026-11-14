"""Read-only ComfyUI node capability checks for selected workflows.

This endpoint deliberately NEVER launches, updates or installs ComfyUI.
Unknown/offline registry state is distinct from an actually missing node.
Avoid returning the full unbounded /object_info payload to browsers.
"""
from __future__ import annotations

import json
import re
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
from fastapi import APIRouter, HTTPException, Query

MAX_REGISTRY_BYTES = 16 * 1024 * 1024
MAX_CLASSES = 64
_CLASS_NAME = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


def parse_requested_classes(comma_classes: str) -> list[str]:
    raw = comma_classes.strip()
    if not raw or len(raw) > 4096:
        raise ValueError("Supply 1–64 ComfyUI node classes.")
    items = list(dict.fromkeys(part.strip() for part in raw.split(",")))
    if len(items) > MAX_CLASSES or not all(_CLASS_NAME.fullmatch(item) for item in items):
        raise ValueError("Invalid ComfyUI node class selection.")
    return items


def summarize_capabilities(
    classes: list[str],
    object_info: dict[str, Any] | None,
) -> dict[str, Any]:
    if object_info is None:
        return {
            "status": "offline",
            "runtime_checked": False,
            "ready": False,
            "requested": classes,
            "available": [],
            "missing": [],
            "unverified": classes,
            "registered_count": 0,
        }
    registered = set(object_info)
    known = [name for name in classes if name in registered]
    missing = [name for name in classes if name not in registered]
    return {
        "status": "ready" if not missing else "missing",
        "runtime_checked": True,
        "ready": not missing,
        "requested": classes,
        "available": known,
        "missing": missing,
        "unverified": [],
        "registered_count": len(registered),
    }


async def read_comfy_registry_readonly(comfy: Any) -> dict[str, Any] | None:
    """Probe the existing process only. Never call ComfyClient.ensure_running."""
    if not await comfy.is_running():
        return None
    async with httpx.AsyncClient(timeout=7.0) as client:
        async with client.stream("GET", f"{comfy.base_url}/object_info") as response:
            response.raise_for_status()
            data = bytearray()
            async for chunk in response.aiter_bytes():
                if len(data) + len(chunk) > MAX_REGISTRY_BYTES:
                    raise ValueError("ComfyUI node registry exceeds inspection budget.")
                data.extend(chunk)
    decoded = json.loads(data)
    if isinstance(decoded, str):
        decoded = json.loads(decoded)
    if not isinstance(decoded, dict):
        raise ValueError("ComfyUI node registry is not an object.")
    return decoded


def make_comfy_capability_router(
    registry_reader: Callable[[], Awaitable[dict[str, Any] | None]],
) -> APIRouter:
    router = APIRouter()

    @router.get("/api/comfy/capabilities")
    async def capabilities(
        classes: str = Query(..., min_length=1, max_length=4096),
    ) -> dict[str, Any]:
        try:
            selected = parse_requested_classes(classes)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        try:
            registry = await registry_reader()
        except (httpx.HTTPError, ValueError, json.JSONDecodeError, OSError):
            # The UI must not claim that nodes are missing when lookup failed.
            report = summarize_capabilities(selected, None)
            report["status"] = "unavailable"
            return report
        return summarize_capabilities(selected, registry)

    return router
