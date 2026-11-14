"""Conservative, bounded static inspection of imported ComfyUI API workflows.

Never calls ComfyUI, downloads models or executes a workflow. Compares saved
node inputs to the running registry when one is supplied by the caller.
Original workflows can be patched by the Nexus generator at dispatch time:
an apparent missing input/choice in saved JSON is a review item, not
proof of a failed generation.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

MAX_WORKFLOW_BYTES = 4 * 1024 * 1024
MAX_WORKFLOW_NODES = 512
MAX_INPUTS_PER_NODE = 128
MAX_FINDINGS = 32


def read_workflow_bounded(path: Path) -> Any:
    with path.open("rb") as stream:
        raw = stream.read(MAX_WORKFLOW_BYTES + 1)
    if len(raw) > MAX_WORKFLOW_BYTES:
        raise ValueError("Workflow exceeds static inspection size limit.")
    return json.loads(raw)


def inspect_workflow_inputs(
    workflow_id: str,
    workflow: Any,
    registry: dict[str, Any] | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "workflow_id": workflow_id,
        "status": "offline" if registry is None else "review",
        "runtime_checked": registry is not None,
        "static_valid": False,
        "ready": False,  # Runtime execution is never certified by this endpoint.
        "node_count": 0,
        "registered_count": len(registry) if registry is not None else 0,
        "missing_classes": 0,
        "broken_links": 0,
        "missing_required_inputs": 0,
        "choice_mismatches": 0,
        "unverified_inputs": 0,
        "findings": [],
        "truncated_findings": False,
        "note": "Saved workflow inspection only; Nexus may patch inputs before generation.",
    }

    def finding(kind: str, node_id: str, node_type: str, input_name: str = "") -> None:
        findings = result["findings"]
        if len(findings) >= MAX_FINDINGS:
            result["truncated_findings"] = True
            return
        findings.append({
            "kind": kind,
            "node_id": node_id[:80],
            "class_type": node_type[:120],
            "input": input_name[:120],
        })

    if not isinstance(workflow, dict):
        result["status"] = "unsupported"
        result["note"] = "Workflow JSON is not an API node mapping."
        return result
    if isinstance(workflow.get("nodes"), list):
        result["status"] = "ui_format"
        result["node_count"] = len(workflow["nodes"])
        result["note"] = "UI workflow widgets require runtime conversion; input signatures cannot be checked statically."
        return result
    nodes = {str(key): value for key, value in workflow.items()
             if isinstance(value, dict) and isinstance(value.get("class_type"), str)}
    if not nodes or len(nodes) > MAX_WORKFLOW_NODES:
        result["status"] = "unsupported"
        result["node_count"] = len(nodes)
        result["note"] = "No API nodes found or workflow exceeds the 512-node inspection budget."
        return result
    result["node_count"] = len(nodes)
    if registry is None:
        return result
    for node_id, node in nodes.items():
        kind = node["class_type"]
        info = registry.get(kind)
        if not isinstance(info, dict):
            result["missing_classes"] += 1
            finding("missing_class", node_id, kind)
            continue
        schema = info.get("input", {})
        if not isinstance(schema, dict):
            result["unverified_inputs"] += 1
            finding("invalid_registry_schema", node_id, kind)
            continue
        required = schema.get("required", {})
        optional = schema.get("optional", {})
        if not isinstance(required, dict) or not isinstance(optional, dict):
            result["unverified_inputs"] += 1
            finding("invalid_registry_schema", node_id, kind)
            continue
        values = node.get("inputs", {})
        if not isinstance(values, dict) or len(values) > MAX_INPUTS_PER_NODE:
            result["unverified_inputs"] += 1
            finding("unsupported_node_inputs", node_id, kind)
            continue
        for name in required:
            if name not in values:
                # Nexus may fill workflow inputs from generation presets at dispatch.
                result["missing_required_inputs"] += 1
                finding("possibly_missing_required", node_id, kind, str(name))
        for name, value in values.items():
            spec = required.get(name, optional.get(name))
            if spec is None:
                result["unverified_inputs"] += 1
                finding("unknown_input", node_id, kind, str(name))
                continue
            expected = spec[0] if isinstance(spec, list) and spec else None
            if (
                isinstance(value, list) and len(value) == 2
                and isinstance(value[0], (str, int))
                and type(value[1]) is int
            ):
                source = nodes.get(str(value[0]))
                if source is None:
                    result["broken_links"] += 1
                    finding("unresolved_link", node_id, kind, str(name))
                    continue
                source_info = registry.get(source["class_type"], {})
                output_types = source_info.get("output") if isinstance(source_info, dict) else None
                # ComfyUI object_info may label output slots with RETURN_TYPES.
                if not isinstance(output_types, list):
                    output_types = source_info.get("RETURN_TYPES") if isinstance(source_info, dict) else None
                if value[1] < 0 or (isinstance(output_types, list) and value[1] >= len(output_types)):
                    result["broken_links"] += 1
                    finding("invalid_output_slot", node_id, kind, str(name))
                elif (
                    isinstance(expected, str) and isinstance(output_types, list)
                    and isinstance(output_types[value[1]], str)
                    and expected != "*" and output_types[value[1]] != "*"
                    and expected != output_types[value[1]]
                ):
                    result["broken_links"] += 1
                    finding("link_type_mismatch", node_id, kind, str(name))
                elif not isinstance(output_types, list):
                    result["unverified_inputs"] += 1
                    finding("unverified_link_type", node_id, kind, str(name))
            elif isinstance(expected, list) and isinstance(value, (str, int, float, bool)):
                if value not in expected:
                    result["choice_mismatches"] += 1
                    finding("possible_choice_mismatch", node_id, kind, str(name))
            elif expected is None:
                result["unverified_inputs"] += 1
                finding("unknown_registry_input", node_id, kind, str(name))
    if result["missing_classes"] or result["broken_links"]:
        result["status"] = "issues"
    elif result["missing_required_inputs"] or result["choice_mismatches"] or result["unverified_inputs"]:
        result["status"] = "review"
    else:
        result["status"] = "static_valid"
        result["static_valid"] = True
    return result
