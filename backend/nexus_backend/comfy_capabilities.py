"""Read-only ComfyUI node capability checks for selected workflows.

This endpoint deliberately NEVER launches, updates or installs ComfyUI.
Unknown/offline registry state is distinct from an actually missing node.
Avoid returning the full unbounded /object_info payload to browsers.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from .workflow_input_audit import inspect_workflow_inputs, read_workflow_bounded

import httpx
from fastapi import APIRouter, HTTPException, Query

MAX_REGISTRY_BYTES = 16 * 1024 * 1024
MAX_CLASSES = 64
MAX_WORKFLOW_CLASSES = 256
_WORKFLOW_ID = re.compile(r"^[A-Za-z0-9_.-]{1,128}$")
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
    *,
    workflow_lookup: Callable[[str], list[str] | None] | None = None,
    workflow_path_lookup: Callable[[str], Path | None] | None = None,
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


    @router.get("/api/comfy/workflow-capabilities")
    async def workflow_capabilities(
        workflow_id: str = Query(..., min_length=1, max_length=128),
    ) -> dict[str, Any]:
        """Check all class types defined in the actual imported workflow JSON."""
        if not _WORKFLOW_ID.fullmatch(workflow_id):
            raise HTTPException(status_code=400, detail="Invalid workflow ID.")
        if workflow_lookup is None:
            raise HTTPException(status_code=404, detail="Workflow inspection unavailable.")
        try:
            raw_classes = workflow_lookup(workflow_id)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise HTTPException(status_code=422, detail="Workflow could not be inspected.") from exc
        if raw_classes is None:
            raise HTTPException(status_code=404, detail="Workflow not found.")
        unique_classes = list(dict.fromkeys(raw_classes))
        valid_classes = [
            item for item in unique_classes
            if isinstance(item, str) and item.strip() and len(item) <= 128
            and not any(ord(char) < 32 for char in item)
        ]
        invalid_count = len(unique_classes) - len(valid_classes)
        if not unique_classes or len(unique_classes) > MAX_WORKFLOW_CLASSES:
            return {
                "status": "unverified",
                "runtime_checked": False,
                "ready": False,
                "requested": valid_classes[:16],
                "available": [],
                "missing": [],
                "unverified": valid_classes[:16],
                "registered_count": 0,
                "workflow_id": workflow_id,
                "required_count": len(unique_classes),
                "invalid_class_count": invalid_count,
                "warning": "No workflow node classes or too many classes for a bounded inspection.",
            }
        try:
            registry = await registry_reader()
        except (httpx.HTTPError, ValueError, json.JSONDecodeError, OSError):
            report = summarize_capabilities(valid_classes, None)
            report["status"] = "unavailable"
        else:
            report = summarize_capabilities(valid_classes, registry)
        report["workflow_id"] = workflow_id
        report["required_count"] = len(unique_classes)
        report["invalid_class_count"] = invalid_count
        if invalid_count:
            report["ready"] = False
            report["status"] = "partial"
            report["warning"] = "Some workflow node identifiers are not inspectable."
        return report


    @router.get("/api/comfy/workflow-inputs")
    async def workflow_input_capabilities(
        workflow_id: str = Query(..., min_length=1, max_length=128),
    ) -> dict[str, Any]:
        """Compare the saved workflow JSON to the running node registry.

        The result is static diagnosis, not end-to-end execution certification.
        """
        if not _WORKFLOW_ID.fullmatch(workflow_id):
            raise HTTPException(status_code=400, detail="Invalid workflow ID.")
        if workflow_path_lookup is None:
            raise HTTPException(status_code=404, detail="Workflow inspection unavailable.")
        try:
            path = await asyncio.to_thread(workflow_path_lookup, workflow_id)
            if path is None:
                raise HTTPException(status_code=404, detail="Workflow not found.")
            data = await asyncio.to_thread(read_workflow_bounded, path)
        except HTTPException:
            raise
        except (OSError, ValueError, json.JSONDecodeError, UnicodeError):
            raise HTTPException(status_code=422, detail="Workflow cannot be inspected safely.")
        try:
            registry = await registry_reader()
        except (httpx.HTTPError, ValueError, json.JSONDecodeError, OSError):
            report = inspect_workflow_inputs(workflow_id, data, None)
            if report["status"] == "offline":
                report["status"] = "unavailable"
            return report
        return inspect_workflow_inputs(workflow_id, data, registry)

    return router
