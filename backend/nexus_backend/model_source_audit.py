"""Bounded, read-only model-file presence checks for patched Comfy API graphs.

Only configured Nexus model roots used by asset resolution are inspected.
No model contents are opened, no symlink is traversed, and no absolute paths
or model filenames are returned to the caller.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .config import DEFAULT_MODEL_SOURCES, NexusSettings

MAX_MODEL_REFERENCES = 24
MAX_MODEL_SOURCE_ROOTS = 64
MAX_SCAN_ENTRIES = 20_000
MAX_SCAN_DEPTH = 8
MAX_MATCHES_PER_REFERENCE = 4

_MODEL_INPUT_CATEGORIES: dict[str, tuple[str, ...]] = {
    "ckpt_name": ("checkpoints",),
    "checkpoint_name": ("checkpoints",),
    "unet_name": ("unet", "diffusion_models"),
    "diffusion_model": ("diffusion_models", "unet"),
    "model_name": ("checkpoints", "diffusion_models", "unet"),
    "vae_name": ("vae",),
    "video_vae_name": ("vae",),
    "audio_vae_name": ("vae",),
    "text_encoder": ("text_encoders", "clip"),
    "clip_name": ("clip", "text_encoders"),
    "clip_name1": ("clip", "text_encoders"),
    "clip_name2": ("clip", "text_encoders"),
    "lora_name": ("loras",),
    "control_net_name": ("controlnet",),
    "controlnet_name": ("controlnet",),
    "upscale_model": ("upscale_models",),
    "latent_upscale_model": ("latent_upscale_models",),
}

_CATEGORY_ALIASES = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "diffusion_model": "diffusion_models",
    "diffusion_models": "diffusion_models",
    "unet": "unet",
    "vae": "vae",
    "audio_vae": "vae",
    "text_encoder": "text_encoders",
    "text_encoders": "text_encoders",
    "clip": "clip",
    "clip_vision": "clip_vision",
    "lora": "loras",
    "loras": "loras",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "model_patch": "model_patches",
    "model_patches": "model_patches",
    "upscale": "upscale_models",
    "upscale_models": "upscale_models",
    "latent_upscale": "latent_upscale_models",
    "latent_upscale_models": "latent_upscale_models",
    "frame_interpolation": "frame_interpolation",
    "video_restore_models": "video_restore_models",
    "denoise_models": "denoise_models",
    "face_restore_models": "face_restore_models",
    "background_removal": "background_removal",
    "embeddings": "embeddings",
    "animatediff_models": "animatediff_models",
    "animatediff_motion_lora": "animatediff_motion_lora",
    "style_models": "style_models",
    "diffusers": "diffusers",
    "3d": "3d",
}
_ALL_CATEGORIES = tuple(DEFAULT_MODEL_SOURCES)
_DEFAULT_EXTENSIONS = (".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx")


def _model_suffix(value: str, extensions: set[str]) -> bool:
    normalized = value.strip().replace("\\", "/")
    return Path(normalized.rsplit("/", 1)[-1]).suffix.lower() in extensions


def _normalize_model_reference(value: str, extensions: set[str]) -> tuple[str, str] | None:
    if not value or len(value) > 512 or any(ord(char) < 32 for char in value):
        return None
    normalized = value.strip().replace("\\", "/")
    if not normalized or normalized.startswith("/") or ":" in normalized:
        return None
    parts = [part for part in normalized.split("/") if part not in {"", "."}]
    if not parts or any(part == ".." for part in parts):
        return None
    relative = "/".join(parts)
    basename = parts[-1]
    if Path(basename).suffix.lower() not in extensions:
        return None
    return basename.casefold(), relative.casefold()


def _entry_metadata(node_id: str, class_type: str, input_name: str, categories: tuple[str, ...]) -> dict[str, Any]:
    return {
        "node_id": str(node_id)[:80],
        "class_type": class_type[:120],
        "input": str(input_name)[:120],
        "categories": list(categories[:4]),
    }


def _extract_references(
    graph: dict[str, dict[str, Any]],
    extensions: set[str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool]:
    references: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str, tuple[str, ...]]] = set()
    truncated = False
    for node_id, node in graph.items():
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        class_type = str(node.get("class_type") or "")[:120]
        for input_name, value in inputs.items():
            if not isinstance(value, str) or not _model_suffix(value, extensions):
                continue
            key = str(input_name).lower()
            categories = _MODEL_INPUT_CATEGORIES.get(key, _ALL_CATEGORIES)
            normalized = _normalize_model_reference(value, extensions)
            if normalized is None:
                skipped.append({
                    **_entry_metadata(node_id, class_type, str(input_name), categories),
                    "status": "unverified",
                    "reason": "unsafe_or_unsupported_reference",
                })
            else:
                basename, relative = normalized
                identity = (str(node_id), key, basename, tuple(categories))
                if identity in seen:
                    continue
                seen.add(identity)
                references.append({
                    **_entry_metadata(node_id, class_type, str(input_name), categories),
                    "basename": basename,
                    "relative": relative,
                })
            if len(references) + len(skipped) >= MAX_MODEL_REFERENCES:
                truncated = True
                return references, skipped, truncated
    return references, skipped, truncated


def _configured_roots(
    settings: NexusSettings,
    categories: tuple[str, ...],
) -> tuple[list[dict[str, Any]], bool]:
    roots: dict[str, dict[str, Any]] = {}
    source_truncated = False

    def add(path: Path, label: str, category: str) -> None:
        key = os.path.normcase(os.path.abspath(os.fspath(path)))
        current = roots.get(key)
        if current is None:
            roots[key] = {"path": path, "labels": {label}, "categories": {category}}
        else:
            current["labels"].add(label)
            current["categories"].add(category)

    for category in categories:
        add(settings.models_dir / category, f"models_dir/{category}", category)

    for alias, paths in settings.model_sources.items():
        normalized = _CATEGORY_ALIASES.get(str(alias).strip().lower(), str(alias).strip().lower())
        if normalized not in categories or not isinstance(paths, list):
            continue
        if len(paths) > MAX_MODEL_SOURCE_ROOTS:
            source_truncated = True
        for index, path in enumerate(paths[:MAX_MODEL_SOURCE_ROOTS]):
            add(Path(path), f"model_sources/{normalized}[{index + 1}]", normalized)

    ordered = sorted(roots.values(), key=lambda item: str(item["path"]).casefold())
    truncated = source_truncated or len(ordered) > MAX_MODEL_SOURCE_ROOTS
    return ordered[:MAX_MODEL_SOURCE_ROOTS], truncated


def _scan_root(
    root: Path,
    labels: set[str],
    wanted: list[dict[str, Any]],
    budget: dict[str, Any],
) -> tuple[dict[int, dict[str, list[str]]], bool, bool]:
    """Return distinct physical file matches by reference index, completeness, scanned."""
    matches: dict[int, dict[str, list[str]]] = {}
    complete = True
    scanned = False

    try:
        if root.is_symlink() or not root.exists():
            return matches, False, False
        root_real = root.resolve(strict=True)
        if not root_real.is_dir():
            return matches, False, False
    except OSError:
        return matches, False, False

    scanned = True
    wanted_by_name: dict[str, list[tuple[int, dict[str, Any]]]] = {}
    for index, reference in enumerate(wanted):
        wanted_by_name.setdefault(reference["basename"], []).append((index, reference))

    stack: list[tuple[Path, int]] = [(root_real, 0)]
    while stack:
        current, depth = stack.pop()
        try:
            if current.is_symlink():
                complete = False
                continue
            resolved_current = current.resolve(strict=True)
            if not resolved_current.is_relative_to(root_real):
                complete = False
                continue
            with os.scandir(resolved_current) as entries:
                for entry in entries:
                    budget["entries"] += 1
                    if budget["entries"] > MAX_SCAN_ENTRIES:
                        budget["exhausted"] = True
                        return matches, False, scanned
                    try:
                        if entry.is_symlink():
                            # Do not inspect the target, even to classify it.
                            complete = False
                            continue
                        if entry.is_dir(follow_symlinks=False):
                            if depth >= MAX_SCAN_DEPTH:
                                complete = False
                                continue
                            stack.append((Path(entry.path), depth + 1))
                            continue
                        if not entry.is_file(follow_symlinks=False):
                            continue
                        candidates = wanted_by_name.get(entry.name.casefold(), [])
                        if not candidates:
                            continue
                        file_real = Path(entry.path).resolve(strict=True)
                        if not file_real.is_relative_to(root_real):
                            complete = False
                            continue
                        relative = file_real.relative_to(root_real).as_posix().casefold()
                        file_identity = os.path.normcase(os.fspath(file_real))
                        for index, reference in candidates:
                            if "/" in reference["relative"] and relative != reference["relative"]:
                                continue
                            by_file = matches.setdefault(index, {})
                            labels_for_file = by_file.setdefault(file_identity, [])
                            for label in sorted(labels):
                                if label not in labels_for_file and len(labels_for_file) < MAX_MATCHES_PER_REFERENCE:
                                    labels_for_file.append(label)
                    except OSError:
                        complete = False
        except OSError:
            complete = False
    return matches, complete, scanned


def audit_model_references(
    graph: dict[str, dict[str, Any]],
    settings: NexusSettings,
) -> dict[str, Any]:
    """Check model filenames against the configured roots without opening model files."""
    extensions = {
        str(item).lower() if str(item).startswith(".") else f".{str(item).lower()}"
        for item in settings.supported_model_extensions
        if str(item).strip()
    } or set(_DEFAULT_EXTENSIONS)
    references, skipped, references_truncated = _extract_references(graph, extensions)
    result: dict[str, Any] = {
        "status": "not_applicable" if not references and not skipped else "partial",
        "references_checked": len(references) + len(skipped),
        "found": 0,
        "multiple_matches": 0,
        "not_found": 0,
        "unverified": len(skipped),
        "roots_checked": 0,
        "entries_scanned": 0,
        "truncated": references_truncated,
        "entries": list(skipped),
        "note": (
            "Read-only filename checks in configured Nexus model roots only. "
            "No model contents were opened; symlinks were not traversed; no files were downloaded."
        ),
    }
    if not references:
        if skipped:
            result["status"] = "partial"
        return result

    all_categories = tuple(dict.fromkeys(
        category for reference in references for category in reference["categories"]
    ))
    roots, roots_truncated = _configured_roots(settings, all_categories)
    budget: dict[str, Any] = {"entries": 0, "exhausted": False}
    found_by_reference: dict[int, dict[str, list[str]]] = {}
    complete_by_reference = {index: not roots_truncated for index in range(len(references))}
    configured_by_reference = {index: 0 for index in range(len(references))}

    for root_info in roots:
        applicable = [
            index for index, reference in enumerate(references)
            if set(reference["categories"]) & root_info["categories"]
        ]
        if not applicable:
            continue
        for index in applicable:
            configured_by_reference[index] += 1
        selected = [references[index] for index in applicable]
        matches, complete, scanned = _scan_root(
            root_info["path"], root_info["labels"], selected, budget,
        )
        if scanned:
            result["roots_checked"] += 1
        if not complete:
            for index in applicable:
                complete_by_reference[index] = False
        for local_index, by_file in matches.items():
            original_index = applicable[local_index]
            current = found_by_reference.setdefault(original_index, {})
            for file_identity, labels in by_file.items():
                current.setdefault(file_identity, [])
                for label in labels:
                    if label not in current[file_identity]:
                        current[file_identity].append(label)
        if budget["exhausted"]:
            for index in range(len(references)):
                if index not in found_by_reference:
                    complete_by_reference[index] = False
            break

    result["entries_scanned"] = budget["entries"]
    result["truncated"] = result["truncated"] or roots_truncated or bool(budget["exhausted"])
    for index, reference in enumerate(references):
        by_file = found_by_reference.get(index, {})
        file_count = len(by_file)
        sources = list(dict.fromkeys(
            label for labels in by_file.values() for label in labels
        ))[:MAX_MATCHES_PER_REFERENCE]
        if file_count > 1:
            status = "multiple_matches"
            result["multiple_matches"] += 1
        elif file_count == 1:
            status = "found"
            result["found"] += 1
        elif configured_by_reference[index] == 0 or not complete_by_reference[index]:
            status = "unverified"
            result["unverified"] += 1
        else:
            status = "not_found"
            result["not_found"] += 1
        result["entries"].append({
            **_entry_metadata(
                reference["node_id"], reference["class_type"], reference["input"],
                tuple(reference["categories"]),
            ),
            "status": status,
            "match_count": file_count,
            "sources": sources,
        })

    if result["unverified"] or result["truncated"]:
        result["status"] = "partial"
    elif result["not_found"]:
        result["status"] = "issues"
    else:
        result["status"] = "complete"
    return result
