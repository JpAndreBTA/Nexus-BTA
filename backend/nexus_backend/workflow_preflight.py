"""Manual, read-only preflight for saved ComfyUI API workflows.

This path does not prepare/upload input files, start Comfy, write settings,
queue prompts, install nodes or persist a patched graph. It uses the same in-memory workflow graph builder as the live
WorkflowRegistry dispatch on a cloned API graph with bounded,
explicitly supplied scalar values, then checks that graph against /object_info.

This is NOT the complete generation dispatch path. No source/mask/LoRA files,
Director frames or other user assets are staged. Treat results as advisory.
"""
from __future__ import annotations

from copy import deepcopy
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .schemas import GenerateRequest
from .workflow_input_audit import (
    MAX_INPUTS_PER_NODE, MAX_WORKFLOW_NODES, inspect_workflow_inputs,
)


class WorkflowPreflightRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    workflow_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9_.-]+$")
    preset: str = Field(default="SD", min_length=1, max_length=64)
    model_name: str = Field(default="", max_length=256)
    prompt: str = Field(default="", max_length=4096)
    negative_prompt: str = Field(default="", max_length=4096)
    width: int = Field(default=1024, ge=64, le=8192)
    height: int = Field(default=576, ge=64, le=8192)
    steps: int = Field(default=25, ge=1, le=200)
    cfg: float = Field(default=7.0, ge=0, le=40, allow_inf_nan=False)
    sampler: str = Field(default="euler_ancestral", min_length=1, max_length=80)
    scheduler: str = Field(default="karras", min_length=1, max_length=80)
    seed: int = Field(default=0, ge=-1, le=4294967295)
    denoise: float = Field(default=1.0, ge=0, le=1, allow_inf_nan=False)


def _graph_snapshot(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(node_id): node
        for node_id, node in graph.items()
        if isinstance(node, dict) and isinstance(node.get("class_type"), str)
    }


def inspect_patched_preview(
    payload: WorkflowPreflightRequest,
    workflow: Any,
    registry: dict[str, Any] | None,
    model_source_auditor: Callable[[dict[str, dict[str, Any]]], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build an in-memory preview with fixed seed and no side effects."""
    base_note = (
        "Advisory template patch only. Nexus may inject runtime assets, uploaded "
        "reference/mask files, LoRAs, and specialized preset workflows at dispatch. "
        "No prompt was submitted and no generation was executed."
    )
    if not isinstance(workflow, dict) or isinstance(workflow.get("nodes"), list):
        return {
            "status": "unsupported", "ready": False, "preview": True,
            "changed_nodes": 0, "changed_input_fields": 0,
            "note": "Preflight supports saved API-format workflows only. " + base_note,
        }
    original = _graph_snapshot(workflow)
    if not original or len(original) > MAX_WORKFLOW_NODES:
        raise ValueError("Workflow exceeds the preflight node budget.")
    if any(
        not isinstance(node.get("inputs", {}), dict)
        or len(node.get("inputs", {})) > MAX_INPUTS_PER_NODE
        for node in original.values()
    ):
        raise ValueError("Workflow has unsupported or oversized input mappings.")

    request = GenerateRequest(
        workflow_id=payload.workflow_id,
        preset=payload.preset,
        model_name=payload.model_name or None,
        prompt=payload.prompt,
        negative_prompt=payload.negative_prompt,
        width=payload.width,
        height=payload.height,
        steps=payload.steps,
        cfg=payload.cfg,
        sampler=payload.sampler,
        scheduler=payload.scheduler,
        # Fixed seed keeps preview deterministic even if generation seed is random.
        seed=0 if payload.seed == -1 else payload.seed,
        denoise=payload.denoise,
    )
    # The shared builder clones the graph; pass a private request copy because
    # the production patcher may normalize/mutate it. Never persist this preview.
    from .workflows import prepare_workflow_graph

    patched = prepare_workflow_graph(
        original,
        request.model_copy(deep=True),
        object_info=registry,
        assets={"primary_model": payload.model_name} if payload.model_name else {},
    )
    if not isinstance(patched, dict) or len(patched) > MAX_WORKFLOW_NODES:
        raise ValueError("Patched workflow exceeds the preflight node budget.")
    patched_nodes = _graph_snapshot(patched)
    if any(
        not isinstance(node.get("inputs", {}), dict)
        or len(node.get("inputs", {})) > MAX_INPUTS_PER_NODE
        for node in patched_nodes.values()
    ):
        raise ValueError("Patched workflow exceeds the preflight input budget.")
    changed_nodes = 0
    changed_fields = 0
    for node_id in set(original) | set(patched_nodes):
        old = original.get(node_id, {})
        new = patched_nodes.get(node_id, {})
        if old != new:
            changed_nodes += 1
        old_inputs = old.get("inputs", {}) if isinstance(old.get("inputs", {}), dict) else {}
        new_inputs = new.get("inputs", {}) if isinstance(new.get("inputs", {}), dict) else {}
        changed_fields += sum(
            1 for name in set(old_inputs) | set(new_inputs)
            if old_inputs.get(name) != new_inputs.get(name)
        )

    report = inspect_workflow_inputs(payload.workflow_id, patched_nodes, registry)
    if model_source_auditor is not None:
        report["model_sources"] = model_source_auditor(patched_nodes)
    report.update({
        "preview": True,
        "ready": False,
        "changed_nodes": changed_nodes,
        "changed_input_fields": changed_fields,
        "note": base_note,
    })
    return report
