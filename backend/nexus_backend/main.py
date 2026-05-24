from __future__ import annotations

import asyncio
import base64
import json
import os
import re
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote

from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asset_resolver import resolve_generation_assets
from .civitai import download_civitai_asset, resolve_civitai_asset, search_civitai_models
from .comfy_client import ComfyClient, extract_outputs
from .config import load_settings
from .dependencies import custom_node_requirements, install_custom_node_dependencies
from .importer import import_resource
from .scanner import ensure_model_tree, scan_custom_nodes, scan_models
from .schemas import (
    DependencyInstallRequest,
    CivitaiDownloadRequest,
    CivitaiResolveRequest,
    CivitaiSearchRequest,
    GenerateRequest,
    GenerateResponse,
    ImportRequest,
    DistilledLoraSelection,
    RuntimeHealth,
    RuntimeOptions,
    WorkflowSaveRequest,
)
from .templates import ensure_templates_file, load_templates
from .workflows import (
    LTX_OMNICINE_DEFAULT_STRENGTH,
    LTX_OMNICINE_LORA_NAME,
    WorkflowRegistry,
    build_basic_anima_workflow,
    build_basic_flux_workflow,
    build_basic_ltx_img2video_workflow,
    build_basic_qwen_image_workflow,
    build_basic_sd_workflow,
    build_basic_wan_i2video_workflow,
    build_basic_zimage_turbo_workflow,
    convert_ui_to_api,
    detect_workflow_format,
    patch_workflow,
)


settings = load_settings()
generation_jobs: dict[str, dict[str, Any]] = {}
download_jobs: dict[str, dict[str, Any]] = {}
generation_lock = asyncio.Lock()
comfy_idle_task: asyncio.Task[None] | None = None

ANSI = {
    "reset": "\033[0m",
    "muted": "\033[90m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}


def _console_generation(job: dict[str, Any], force: bool = False) -> None:
    progress = int(job.get("progress") or 0)
    bucket = progress // 5
    if not force and job.get("_last_bucket") == bucket and job.get("_last_status") == job.get("status"):
        return
    job["_last_bucket"] = bucket
    job["_last_status"] = job.get("status")
    filled = max(0, min(20, round(progress / 5)))
    bar = "#" * filled + "-" * (20 - filled)
    status = str(job.get("status") or "queued").upper()
    color = ANSI["green"] if status == "COMPLETED" else ANSI["red"] if status == "FAILED" else ANSI["cyan"]
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = str(job.get("message") or "").strip()
    node = str(job.get("node") or "").strip()
    step = ""
    if job.get("current_step") is not None and job.get("total_steps"):
        step = f" step {job['current_step']}/{job['total_steps']}"
    node_part = f" node {node}" if node else ""
    try:
        print(
            f"{ANSI['muted']}[{timestamp}]{ANSI['reset']} {ANSI['red']}NEXUS{ANSI['reset']} "
            f"{color}{status:<9}{ANSI['reset']} {color}{bar}{ANSI['reset']} "
            f"{ANSI['white']}{progress:3d}%{ANSI['reset']}{step}{node_part} {ANSI['muted']}{message}{ANSI['reset']}",
            flush=True,
        )
    except OSError:
        pass


def _update_generation_job(job_id: str, update: dict[str, Any], *, force: bool = False) -> None:
    job = generation_jobs.get(job_id)
    if not job:
        return
    if str(job.get("status") or "").lower() == "cancelled" and str(update.get("status") or "").lower() != "cancelled":
        return
    job.update({key: value for key, value in update.items() if value is not None})
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _console_generation(job, force=force)


def _update_download_job(job_id: str, update: dict[str, Any]) -> None:
    job = download_jobs.get(job_id)
    if not job:
        return
    if update.get("progress") is None and update.get("bytes_downloaded"):
        total = update.get("bytes_total") or job.get("bytes_total") or 0
        if total:
            update["progress"] = round((float(update["bytes_downloaded"]) / float(total)) * 100, 2)
    job.update({key: value for key, value in update.items() if value is not None})
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")


def _write_input_data_image(value: str, prefix: str) -> str:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    match = re.match(r"data:image/([a-zA-Z0-9.+-]+);base64,(.+)", value, flags=re.DOTALL)
    if not match:
        raise ValueError("Invalid image data URL.")
    ext = "jpg" if match.group(1).lower() in {"jpeg", "jpg"} else "png"
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"
    target = settings.input_dir / filename
    target.write_bytes(base64.b64decode(match.group(2)))
    return filename


def _write_input_data_audio(value: str, prefix: str) -> str:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    match = re.match(r"data:audio/([a-zA-Z0-9.+-]+);base64,(.+)", value, flags=re.DOTALL)
    if not match:
        raise ValueError("Invalid audio data URL.")
    mime = match.group(1).lower()
    ext = {
        "mpeg": "mp3",
        "mp3": "mp3",
        "wav": "wav",
        "wave": "wav",
        "x-wav": "wav",
        "flac": "flac",
        "ogg": "ogg",
        "webm": "webm",
        "mp4": "m4a",
        "aac": "aac",
    }.get(mime, "wav")
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"
    target = settings.input_dir / filename
    target.write_bytes(base64.b64decode(match.group(2)))
    return filename


def _write_input_audio_bytes(payload: bytes, prefix: str, ext: str = "wav") -> str:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    clean_ext = re.sub(r"[^a-zA-Z0-9]+", "", ext or "wav").lower()[:8] or "wav"
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.{clean_ext}"
    target = settings.input_dir / filename
    target.write_bytes(payload)
    return filename


def _write_input_data_video(value: str, prefix: str) -> str:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    match = re.match(r"data:video/([a-zA-Z0-9.+-]+);base64,(.+)", value, flags=re.DOTALL)
    if not match:
        raise ValueError("Invalid video data URL.")
    mime = match.group(1).lower()
    ext = {
        "mp4": "mp4",
        "quicktime": "mov",
        "webm": "webm",
        "x-matroska": "mkv",
        "matroska": "mkv",
        "mpeg": "mpg",
    }.get(mime, "mp4")
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"
    target = settings.input_dir / filename
    target.write_bytes(base64.b64decode(match.group(2)))
    return filename


def _audio_key(value: object) -> str:
    text = str(value or "").strip().replace("\\", "/")
    return text.rsplit("/", 1)[-1].lower()


def _materialize_ltx_director_audio(prompt: dict[str, Any]) -> None:
    replacements: dict[str, str] = {}
    missing_custom_audio: list[str] = []
    for node in prompt.values():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "ltxdirector":
            continue
        inputs = node.setdefault("inputs", {})
        custom_audio_enabled = str(inputs.get("use_custom_audio", "")).strip().lower() not in {"", "false", "0", "off", "none", "no"}
        raw_timeline = inputs.get("timeline_data")
        if not isinstance(raw_timeline, str) or not raw_timeline.strip():
            continue
        try:
            timeline = json.loads(raw_timeline)
        except json.JSONDecodeError:
            continue
        changed = False
        segments = timeline.get("segments")
        if isinstance(segments, list):
            for segment in segments:
                if not isinstance(segment, dict):
                    continue
                video_b64 = str(segment.get("videoB64") or "")
                if not video_b64.startswith("data:video/"):
                    continue
                filename = _write_input_data_video(video_b64, "nexus_director_video")
                segment["videoFile"] = filename
                segment["fileName"] = segment.get("fileName") or filename
                segment["videoB64"] = ""
                load_video = segment.setdefault("loadVideo", {})
                if isinstance(load_video, dict):
                    load_video["video"] = filename
                changed = True

        audio_segments = timeline.get("audioSegments")
        if not isinstance(audio_segments, list):
            if changed:
                inputs["timeline_data"] = json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))
            continue
        for segment in audio_segments:
            if not isinstance(segment, dict):
                continue
            audio_b64 = str(segment.get("audioB64") or "")
            old_names = [
                segment.get("audioFile"),
                segment.get("fileName"),
                segment.get("title"),
            ]
            filename = ""
            if audio_b64.startswith("data:audio/"):
                filename = _write_input_data_audio(audio_b64, "nexus_director_audio")
            elif audio_b64 and audio_b64.lower() not in {"embedded", "none", "null"}:
                try:
                    payload = base64.b64decode(audio_b64.split(",", 1)[-1], validate=True)
                except Exception:
                    payload = b""
                if payload:
                    source_name = str(segment.get("audioFile") or segment.get("fileName") or "")
                    ext = Path(source_name).suffix.lstrip(".") or "wav"
                    filename = _write_input_audio_bytes(payload, "nexus_director_audio", ext)
            if filename:
                for old_name in old_names:
                    if old_name:
                        replacements[str(old_name).strip().lower()] = filename
                        replacements[_audio_key(old_name)] = filename
                segment["audioFile"] = filename
                segment["fileName"] = filename
                segment["audioB64"] = ""
                changed = True
                continue

            if custom_audio_enabled:
                current_name = str(segment.get("audioFile") or segment.get("fileName") or "").strip()
                if not current_name:
                    missing_custom_audio.append(str(segment.get("id") or "audio segment"))
                    continue
                input_path = (settings.input_dir / current_name).resolve()
                try:
                    input_path.relative_to(settings.input_dir.resolve())
                except ValueError:
                    missing_custom_audio.append(current_name)
                    continue
                if not input_path.exists():
                    missing_custom_audio.append(current_name)
        if changed:
            inputs["timeline_data"] = json.dumps(timeline, ensure_ascii=False, separators=(",", ":"))

    if missing_custom_audio:
        names = ", ".join(sorted({Path(item).name for item in missing_custom_audio})[:5])
        raise ValueError(
            "LTX Director custom audio is enabled, but the selected audio clip is no longer embedded or present "
            f"in the ComfyUI input folder: {names}. Re-add the audio clip before rendering."
        )

    if not replacements:
        return
    fallback = next(iter(replacements.values())) if len(set(replacements.values())) == 1 else None
    for node in prompt.values():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "loadaudioui":
            continue
        inputs = node.setdefault("inputs", {})
        current = inputs.get("audio")
        replacement = replacements.get(str(current or "").strip().lower()) or replacements.get(_audio_key(current))
        if replacement:
            inputs["audio"] = replacement
        elif fallback and not current:
            inputs["audio"] = fallback


def _normalize_lora_key(value: object) -> str:
    text = str(value or "").strip().replace("/", "\\").lower()
    for prefix in ("models\\loras\\", "loras\\", ".\\models\\loras\\"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
            break
    if text.startswith("ltx2\\"):
        text = "ltx\\" + text.split("\\", 1)[1]
    if "\\" not in text and text.startswith(("ltx", "singularity")):
        text = f"ltx\\{text}"
    return text


def _ensure_ltx_default_distilled_loras(request: GenerateRequest, assets: dict[str, str]) -> None:
    if request.preset.lower() != "ltx":
        return
    existing = {
        _normalize_lora_key(getattr(item, "name", ""))
        for item in request.distilled_loras
        if _normalize_lora_key(getattr(item, "name", ""))
        and _normalize_lora_key(getattr(item, "name", "")) not in {"none", "automatic", "auto"}
    }
    additions: list[DistilledLoraSelection] = []
    default_strengths = {
        "distilled_lora_1": 0.35,
        "distilled_lora_2": 0.50,
    }
    for key in ("distilled_lora_1", "distilled_lora_2"):
        name = assets.get(key)
        normalized = _normalize_lora_key(name)
        if not normalized or normalized in existing:
            continue
        existing.add(normalized)
        additions.append(DistilledLoraSelection(name=name, strength=default_strengths[key]))
    video_options = request.video or {}
    omnicine_enabled = video_options.get("omnicine_enabled", False)
    if isinstance(omnicine_enabled, str):
        omnicine_enabled = omnicine_enabled.lower() not in {"false", "0", "off", "none", "no"}
    if omnicine_enabled is not False:
        omni_name = str(video_options.get("omnicine_lora") or LTX_OMNICINE_LORA_NAME)
        if not _is_omnicine_lora_name(omni_name):
            omni_name = ""
        normalized = _normalize_lora_key(omni_name)
        if normalized and normalized not in existing:
            existing.add(normalized)
            additions.append(DistilledLoraSelection(name=omni_name, strength=LTX_OMNICINE_DEFAULT_STRENGTH))
    if additions:
        request.distilled_loras.extend(additions)


def _is_omnicine_lora_name(value: object) -> bool:
    return any(token in str(value or "").lower() for token in ("omnicine", "singularity"))


def _ensure_wan_4step_loras(request: GenerateRequest, assets: dict[str, str]) -> None:
    if request.preset.lower() != "wan":
        return
    high_lora = assets.get("wan_4step_high_lora")
    low_lora = assets.get("wan_4step_low_lora")
    if not high_lora or not low_lora:
        return
    existing = {
        _normalize_lora_key(
            item.get("relative_name")
            or item.get("relative_path")
            or item.get("lora_name")
            or item.get("name")
            or ""
        )
        for item in request.loras
        if isinstance(item, dict)
    }
    additions: list[dict[str, Any]] = []
    for name, role in ((high_lora, "wan_4step_high"), (low_lora, "wan_4step_low")):
        normalized = _normalize_lora_key(name)
        if not normalized or normalized in existing:
            continue
        existing.add(normalized)
        additions.append(
            {
                "name": name,
                "relative_name": name,
                "strength": 1.0,
                "strength_model": 1.0,
                "strength_clip": 0.0,
                "role": role,
                "auto": True,
            }
        )
    if additions:
        request.loras.extend(additions)


def _output_relative_from_url(value: str) -> str:
    relative = value.split("/outputs/", 1)[1].split("?", 1)[0].split("#", 1)[0]
    return unquote(relative).lstrip("/\\")


def _prepare_reference_value(value: str, prefix: str = "nexus_reference") -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("Reference image could not be resolved.")

    if value.startswith("data:image/"):
        return _write_input_data_image(value, prefix)

    source: Path | None = None
    if value.startswith("/outputs/") or "/outputs/" in value:
        relative = _output_relative_from_url(value)
        source = (settings.output_dir / relative).resolve()
    else:
        candidate = Path(value)
        if candidate.exists():
            source = candidate.resolve()

    if not source or not source.exists():
        raise ValueError("Reference image could not be resolved.")

    suffix = source.suffix.lower() if source.suffix else ".png"
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}{suffix}"
    target = settings.input_dir / filename
    shutil.copy2(source, target)
    return filename


def _reference_image_values(request: GenerateRequest) -> list[str]:
    values: list[str] = []
    if request.img2img.reference_image:
        values.append(request.img2img.reference_image)
    values.extend(request.img2img.reference_images or [])
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = (value or "").strip()
        if normalized and normalized not in seen:
            seen.add(normalized)
            unique.append(normalized)
    return unique


def _prepare_reference_images(request: GenerateRequest) -> list[str]:
    if request.activity != "img2img":
        return []
    values = _reference_image_values(request)[:3]
    return [_prepare_reference_value(value, f"nexus_reference_{index + 1}") for index, value in enumerate(values)]


def _available_comfy_node(object_info: dict[str, Any], *names: str) -> str | None:
    available = set(object_info or {})
    for name in names:
        if name in available:
            return name
    return None


def _prepare_reference_image(request: GenerateRequest) -> str | None:
    value = (request.img2img.reference_image or "").strip()
    if request.activity != "img2img" or not value:
        return None
    return _prepare_reference_value(value, "nexus_reference")


def _prepare_mask_image(request: GenerateRequest) -> str | None:
    value = (request.img2img.mask_image or "").strip()
    mode = request.img2img.mode.lower()
    if request.activity != "img2img" or not value or "inpaint" not in mode:
        return None
    if not value.startswith("data:image/"):
        raise ValueError("Invalid inpaint mask image.")
    return _write_input_data_image(value, "nexus_mask")


def _prepare_controlnet_image(request: GenerateRequest) -> str | None:
    if not request.controlnet.enabled:
        return None
    value = (request.controlnet.image or "").strip()
    if not value:
        values = _reference_image_values(request)
        value = values[0] if values else ""
    if not value:
        return None

    if value.startswith("data:image/"):
        return _write_input_data_image(value, "nexus_controlnet")

    source: Path | None = None
    if value.startswith("/outputs/") or "/outputs/" in value:
        relative = _output_relative_from_url(value)
        source = (settings.output_dir / relative).resolve()
    else:
        candidate = Path(value)
        if candidate.exists():
            source = candidate.resolve()

    if not source or not source.exists():
        raise ValueError("ControlNet image could not be resolved.")

    suffix = source.suffix.lower() if source.suffix else ".png"
    filename = f"nexus_controlnet_{uuid.uuid4().hex[:10]}{suffix}"
    target = settings.input_dir / filename
    shutil.copy2(source, target)
    return filename


def _enriched_lora_metadata(request: GenerateRequest) -> list[dict[str, Any]]:
    try:
        catalog = scan_models(settings)
        lora_index: dict[str, Any] = {}
        for item in catalog.categories.get("loras", []):
            keys = {
                item.name.replace("/", "\\").lower(),
                item.relative_path.replace("/", "\\").lower(),
            }
            relative = item.relative_path.replace("/", "\\")
            if relative.lower().startswith("loras\\"):
                keys.add(relative.split("\\", 1)[1].lower())
            for key in keys:
                lora_index[key] = item
    except Exception:
        lora_index = {}

    enriched: list[dict[str, Any]] = []
    for entry in request.loras:
        if not isinstance(entry, dict):
            continue
        data = dict(entry)
        raw_name = data.get("relative_name") or data.get("relative_path") or data.get("lora_name") or data.get("name") or ""
        lookup = str(raw_name).replace("/", "\\").lower()
        item = lora_index.get(lookup) or lora_index.get(Path(str(raw_name)).name.lower())
        if item:
            data.update(
                {
                    "name": data.get("name") or item.name,
                    "relative_name": data.get("relative_name") or item.relative_path.replace("/", "\\").split("\\", 1)[-1],
                    "relative_path": item.relative_path,
                    "folder": item.folder,
                    "tags": item.tags,
                    "preview": item.preview,
                    "source": item.source,
                    "size_bytes": item.size_bytes,
                    "modified": item.modified,
                }
            )
        enriched.append(data)
    return enriched


def _sanitize_embedded_media_metadata(value: Any) -> Any:
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        for key, item in value.items():
            if key == "timeline_data_json" and isinstance(item, str):
                try:
                    parsed = json.loads(item)
                    sanitized[key] = json.dumps(
                        _sanitize_embedded_media_metadata(parsed),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                except Exception:
                    sanitized[key] = "embedded" if "data:" in item else item
                continue
            if key in {"imageB64", "audioB64", "videoB64", "imageSrc", "videoSrc"} and isinstance(item, str):
                sanitized[key] = "embedded" if item.startswith("data:") else item
                continue
            sanitized[key] = _sanitize_embedded_media_metadata(item)
        return sanitized
    if isinstance(value, list):
        return [_sanitize_embedded_media_metadata(item) for item in value]
    if isinstance(value, str) and value.startswith(("data:image/", "data:audio/", "data:video/")):
        return "embedded"
    return value


def _generation_metadata(request: GenerateRequest, assets: dict[str, Any] | None = None) -> dict[str, Any]:
    assets = assets or {}
    controlnet = request.controlnet.model_dump()
    if controlnet.get("image"):
        image_value = str(controlnet.get("image") or "")
        controlnet["image"] = "embedded" if image_value.startswith("data:image/") else Path(image_value).name
    if assets.get("controlnet_model") and str(controlnet.get("model") or "").strip().lower() in {"", "automatic", "auto"}:
        controlnet["model"] = assets["controlnet_model"]
    video = dict(request.video or {})
    for source_key, target_key in (
        ("video_vae", "video_vae"),
        ("audio_vae", "audio_vae"),
        ("preview_vae", "preview_vae"),
        ("latent_upscale", "latent_upscale"),
        ("detailer_lora", "detailer_lora"),
        ("ic_lora", "ic_lora"),
        ("wan_high_model", "wan_high_model"),
        ("wan_low_model", "wan_low_model"),
        ("wan_4step_high_lora", "wan_4step_high_lora"),
        ("wan_4step_low_lora", "wan_4step_low_lora"),
    ):
        current_value = str(video.get(target_key) or "").strip().lower()
        if assets.get(source_key) and (not current_value or current_value in {"automatic", "auto"}):
            video[target_key] = assets[source_key]
    return {
        "prompt": request.prompt,
        "negative": request.negative_prompt,
        "model": request.model_name or assets.get("primary_model") or Path(request.model_path or "").name,
        "seed": request.seed,
        "steps": request.steps,
        "cfg": request.cfg,
        "sampler": request.sampler,
        "scheduler": request.scheduler,
        "preset": request.preset,
        "activity": request.activity,
        "width": request.width,
        "height": request.height,
        "workflow_id": request.workflow_id or "Default",
        "loras": _enriched_lora_metadata(request),
        "distilled_loras": [item.model_dump(mode="json") for item in request.distilled_loras],
        "video": video,
        "director": _sanitize_embedded_media_metadata(request.director),
        "vae": assets.get("vae") or assets.get("video_vae") or request.vae,
        "text_encoder": assets.get("text_encoder") or request.text_encoder,
        "controlnet": controlnet,
    }


def _annotate_output_metadata(outputs: list[dict[str, Any]], request: GenerateRequest, assets: dict[str, Any] | None = None) -> None:
    metadata = _generation_metadata(request, assets)
    for output in outputs:
        relative = str(output.get("path") or output.get("filename") or "")
        if not relative:
            continue
        path = (settings.output_dir / relative).resolve()
        if not path.exists() or not path.is_relative_to(settings.output_dir.resolve()):
            continue
        file_metadata = {
            **metadata,
            "file": path.name,
            "path": relative.replace("\\", "/"),
            "kind": output.get("kind") or path.suffix.lower().lstrip("."),
        }
        try:
            path.with_suffix(path.suffix + ".nexus.json").write_text(json.dumps(file_metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
        if path.suffix.lower() != ".png":
            continue
        try:
            from PIL import Image
            from PIL.PngImagePlugin import PngInfo

            with Image.open(path) as image:
                pnginfo = PngInfo()
                for key, value in image.text.items():
                    pnginfo.add_text(key, value)
                pnginfo.add_text("nexus_bta", json.dumps(file_metadata, ensure_ascii=False))
                image.save(path, pnginfo=pnginfo)
        except Exception:
            continue


def _recent_output_files(start_timestamp: float, limit: int = 8) -> list[dict[str, Any]]:
    if not settings.output_dir.exists():
        return []
    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mkv", ".mov", ".avi"}
    root = settings.output_dir.resolve()
    candidates: list[Path] = []
    for path in settings.output_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in media_suffixes:
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(root):
                continue
            if path.stat().st_mtime + 2 < start_timestamp:
                continue
        except Exception:
            continue
        candidates.append(path)
    outputs: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True)[:limit]:
        relative = path.relative_to(settings.output_dir).as_posix()
        suffix = path.suffix.lower()
        kind = "video" if suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"} else "image"
        outputs.append(
            {
                "kind": kind,
                "filename": path.name,
                "subfolder": "" if path.parent == settings.output_dir else path.parent.relative_to(settings.output_dir).as_posix(),
                "type": "output",
                "path": relative,
                "url": f"/outputs/{quote(relative, safe='/')}",
            }
        )
    return outputs


async def _recover_outputs_from_history(prompt_id: str | None, start_timestamp: float) -> list[dict[str, Any]]:
    if prompt_id:
        for _ in range(6):
            try:
                history = await comfy.history(prompt_id)
                outputs = extract_outputs(history.get(prompt_id, {}))
                if outputs:
                    return outputs
            except Exception:
                pass
            await asyncio.sleep(1)
    return _recent_output_files(start_timestamp)


def _read_output_metadata(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".nexus.json")
    if sidecar.exists():
        try:
            parsed = json.loads(sidecar.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass
    if path.suffix.lower() != ".png":
        return {}
    try:
        from PIL import Image

        with Image.open(path) as image:
            raw = image.text.get("nexus_bta") if hasattr(image, "text") else None
            if raw:
                parsed = json.loads(raw)
                return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}
    return {}


def _cleanup_generation_temp() -> None:
    for pattern in ("nexus_reference_*", "nexus_mask_*", "nexus_controlnet_*", "nexus_director_audio_*"):
        for path in settings.input_dir.glob(pattern):
            if path.is_file():
                path.unlink(missing_ok=True)
    if settings.temp_dir.exists():
        for item in settings.temp_dir.iterdir():
            if item.name == "loaded_workflows":
                continue
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
            else:
                item.unlink(missing_ok=True)
    _cleanup_local_app_temp()
    cleanup_embedded_comfy_artifacts()


def _cleanup_local_app_temp() -> None:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        return
    temp_root = Path(local_appdata) / "Temp"
    if not temp_root.exists():
        return

    safe_patterns = (
        "nexus_*",
        "NexusBTA*",
        "nexus-bta*",
        "ComfyUI*",
        "comfyui*",
    )
    for pattern in safe_patterns:
        for path in temp_root.glob(pattern):
            try:
                resolved = path.resolve()
            except OSError:
                continue
            if not resolved.is_relative_to(temp_root.resolve()):
                continue
            if resolved.is_dir():
                shutil.rmtree(resolved, ignore_errors=True)
            elif resolved.is_file():
                resolved.unlink(missing_ok=True)


def cleanup_embedded_comfy_artifacts() -> None:
    """Keep copied ComfyUI runtime clean; Nexus owns output/temp/user paths."""
    comfy_root = settings.comfy_root.resolve()
    if not comfy_root.exists():
        return

    for name in ("output", "temp"):
        path = (comfy_root / name).resolve()
        if path.exists() and path.is_relative_to(comfy_root):
            shutil.rmtree(path, ignore_errors=True)

    for log_file in list(comfy_root.glob("*.log")) + list((comfy_root / "user").rglob("*.log") if (comfy_root / "user").exists() else []):
        resolved = log_file.resolve()
        if resolved.is_relative_to(comfy_root):
            resolved.unlink(missing_ok=True)


cleanup_embedded_comfy_artifacts()
ensure_templates_file()
ensure_model_tree(settings)
workflow_registry = WorkflowRegistry(settings)
comfy = ComfyClient(settings)


def _canonical_vram_policy(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text in {"low", "lowvram"}:
        return "low"
    if text in {"med", "medium", "medvram", "normal", "balanced", "balance"}:
        return "balanced"
    if text in {"high", "highvram"}:
        return "high"
    if text.startswith("cpu"):
        return "cpu"
    return "balanced"


def _canonical_attention_backend(value: str | None) -> str:
    text = str(value or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if text in {"sage", "sageattention"}:
        return "sage"
    if text in {"flash", "flashattention"}:
        return "flash"
    if text in {"pytorch", "pytorchsdpa", "sdpa"}:
        return "pytorch"
    return "auto"


def _canonical_precision(value: str | None) -> str:
    text = str(value or "").strip().lower().replace(" ", "").replace("_", "-")
    return text if text in {"auto", "fp16", "fp32", "bf16", "fp8"} else "auto"


def _apply_runtime_options(options: RuntimeOptions | None) -> bool:
    if not options:
        return False
    next_vram = _canonical_vram_policy(options.vram_policy)
    next_attention = _canonical_attention_backend(options.attention_backend)
    next_precision = _canonical_precision(options.precision)
    next_disable_xformers = bool(options.disable_xformers)
    changed = (
        _canonical_vram_policy(settings.runtime.vram_policy) != next_vram
        or _canonical_attention_backend(settings.runtime.attention_backend) != next_attention
        or _canonical_precision(settings.runtime.precision) != next_precision
        or bool(settings.runtime.disable_xformers) != next_disable_xformers
    )
    settings.runtime.vram_policy = next_vram
    settings.runtime.attention_backend = next_attention
    settings.runtime.precision = next_precision
    settings.runtime.disable_xformers = next_disable_xformers
    settings.runtime.enable_sage_attention = next_attention == "sage"
    settings.runtime.enable_flash_attention = next_attention == "flash"
    return changed


async def _optional_comfy_object_info() -> dict[str, Any]:
    try:
        if not await comfy.is_running():
            return {}
        return await comfy.object_info()
    except Exception:
        return {}


def _active_generation_jobs() -> list[dict[str, Any]]:
    active_statuses = {"queued", "starting", "preparing", "building", "running", "polling"}
    return [job for job in generation_jobs.values() if str(job.get("status") or "").lower() in active_statuses]


def _generation_queue_position(job_id: str) -> int:
    active_statuses = {"queued", "starting", "preparing", "building", "running", "polling"}
    ordered = [
        job
        for job in generation_jobs.values()
        if str(job.get("status") or "").lower() in active_statuses
    ]
    ordered.sort(key=lambda job: str(job.get("created_at") or ""))
    for index, job in enumerate(ordered, start=1):
        if job.get("job_id") == job_id:
            return index
    return 0


async def _guard_runtime_mutation(action: str, force: bool = False) -> None:
    active = _active_generation_jobs()
    if not active:
        return
    if not force:
        raise HTTPException(
            status_code=409,
            detail=f"Cannot {action} while {len(active)} generation job(s) are active. Cancel the job or retry with force=true.",
        )
    for job in active:
        job.update(
            {
                "status": "cancelled",
                "progress": 100,
                "message": f"Generation cancelled before ComfyUI {action}.",
                "error": "cancelled",
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        prompt_id = job.get("prompt_id")
        if prompt_id:
            try:
                await comfy.interrupt(str(prompt_id))
            except Exception:
                pass
        _console_generation(job, force=True)


def _cancel_comfy_idle_release() -> None:
    global comfy_idle_task
    if comfy_idle_task and not comfy_idle_task.done():
        comfy_idle_task.cancel()
    comfy_idle_task = None


def _schedule_comfy_idle_release() -> None:
    global comfy_idle_task
    _cancel_comfy_idle_release()
    unload_delay = max(0, int(getattr(settings.runtime, "idle_unload_seconds", 90) or 0))
    stop_delay = max(0, int(getattr(settings.runtime, "idle_stop_seconds", 300) or 0))
    if not unload_delay and not stop_delay:
        return
    comfy_idle_task = asyncio.create_task(_comfy_idle_release_worker(unload_delay, stop_delay))


async def _comfy_idle_release_worker(unload_delay: int, stop_delay: int) -> None:
    try:
        if unload_delay:
            await asyncio.sleep(unload_delay)
        if not _active_generation_jobs() and await comfy.is_running():
            await comfy.free_memory(unload_models=True, free_memory=True)
            cleanup_embedded_comfy_artifacts()
        if stop_delay and stop_delay > unload_delay:
            await asyncio.sleep(stop_delay - unload_delay)
            if not _active_generation_jobs() and await comfy.is_running():
                comfy.stop()
                cleanup_embedded_comfy_artifacts()
    except asyncio.CancelledError:
        raise
    except Exception:
        return


async def _release_comfy_memory_if_idle() -> None:
    if _active_generation_jobs():
        return
    try:
        if await comfy.is_running():
            await comfy.free_memory(unload_models=True, free_memory=True)
            cleanup_embedded_comfy_artifacts()
    except Exception:
        return


def _process_snapshot(pid: int | None) -> dict[str, Any]:
    if not pid:
        return {}
    try:
        import psutil

        process = psutil.Process(pid)
        with process.oneshot():
            return {
                "pid": pid,
                "name": process.name(),
                "status": process.status(),
                "memory_bytes": process.memory_info().rss,
                "cpu_percent": process.cpu_percent(interval=0.0),
            }
    except Exception:
        return {"pid": pid}


def _comfy_port_owner_pid() -> int | None:
    try:
        import psutil

        for conn in psutil.net_connections(kind="inet"):
            if conn.status == "LISTEN" and conn.laddr and getattr(conn.laddr, "port", None) == settings.runtime.comfy_port:
                return conn.pid
    except Exception:
        return None
    return None


async def _runtime_memory_snapshot() -> dict[str, Any]:
    stats = await comfy.system_stats()
    comfy_pid = _comfy_port_owner_pid() or (comfy.process.pid if comfy.process and comfy.process.poll() is None else None)
    return {
        "comfy_running": bool(stats),
        "comfy_url": comfy.base_url,
        "comfy_process": _process_snapshot(comfy_pid),
        "backend_process": _process_snapshot(os.getpid()),
        "comfy_system_stats": stats,
        "idle_unload_seconds": getattr(settings.runtime, "idle_unload_seconds", 90),
        "idle_stop_seconds": getattr(settings.runtime, "idle_stop_seconds", 300),
        "active_generation_jobs": len(_active_generation_jobs()),
    }

app = FastAPI(title="Nexus BTA Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if settings.output_dir.exists():
    app.mount("/outputs", StaticFiles(directory=settings.output_dir), name="outputs")
if settings.models_dir.exists():
    app.mount("/model-assets", StaticFiles(directory=settings.models_dir), name="model-assets")
assets_dir = settings.project_root / "assets"
if assets_dir.exists():
    app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")


@app.on_event("startup")
async def startup() -> None:
    settings.ensure_directories()
    cleanup_embedded_comfy_artifacts()
    ensure_model_tree(settings)
    workflow_registry.import_reference_workflows()


@app.get("/api/health", response_model=RuntimeHealth)
async def health() -> RuntimeHealth:
    return RuntimeHealth(
        comfy_running=await comfy.is_running(),
        comfy_url=comfy.base_url,
        comfy_root_exists=settings.comfy_root.exists(),
        comfy_python_exists=settings.comfy_python.exists(),
        models_dir=str(settings.models_dir),
        custom_nodes_dir=str(settings.custom_nodes_dir),
    )


@app.get("/api/config")
async def config() -> dict[str, Any]:
    return settings.model_dump(mode="json")


@app.get("/api/templates")
async def templates() -> dict[str, Any]:
    return load_templates()


@app.post("/api/model-tree")
async def model_tree() -> dict[str, Any]:
    return {"created": ensure_model_tree(settings)}


@app.get("/api/models")
async def models(include_references: bool = Query(False)) -> dict[str, Any]:
    return scan_models(settings, include_references=include_references).model_dump(mode="json")


@app.get("/api/loras")
async def loras(include_references: bool = Query(False)) -> list[dict[str, Any]]:
    catalog = scan_models(settings, include_references=include_references)
    return [item.model_dump(mode="json") for item in catalog.categories.get("loras", [])]


@app.get("/api/custom-nodes")
async def custom_nodes(include_references: bool = Query(False)) -> list[dict[str, Any]]:
    return [
        node.model_dump(mode="json")
        for node in scan_custom_nodes(settings, include_references=include_references)
    ]


@app.get("/api/custom-nodes/dependencies")
async def custom_node_dependencies() -> dict[str, str]:
    return {name: str(path) for name, path in custom_node_requirements(settings).items()}


@app.post("/api/custom-nodes/install-dependencies")
async def install_dependencies(request: DependencyInstallRequest) -> dict[str, Any]:
    installed, errors = install_custom_node_dependencies(
        settings,
        node_names=request.node_names,
        all_enabled=request.all_enabled,
    )
    return {"installed": installed, "errors": errors}


@app.get("/api/workflows")
async def workflows() -> list[dict[str, Any]]:
    return [workflow.model_dump(mode="json") for workflow in workflow_registry.list_workflows()]


@app.post("/api/workflows/import")
async def import_workflow(
    file: UploadFile = File(...),
    install_dependencies: bool = Query(False),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Workflow must be a .json file.")
    content = await file.read()
    try:
        workflow = workflow_registry.import_workflow_file(file.filename, content)
        object_info = await _optional_comfy_object_info()
        analysis = workflow_registry.analyze_workflow(
            workflow,
            object_info=object_info,
            install_dependencies=install_dependencies,
        )
        return analysis.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workflows/load")
async def load_workflow(
    file: UploadFile = File(...),
    install_dependencies: bool = Query(False),
) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".json"):
        raise HTTPException(status_code=400, detail="Workflow must be a .json file.")
    content = await file.read()
    try:
        workflow = workflow_registry.load_workflow_file(file.filename, content)
        object_info = await _optional_comfy_object_info()
        analysis = workflow_registry.analyze_workflow(
            workflow,
            object_info=object_info,
            install_dependencies=install_dependencies,
        )
        return analysis.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/workflows/save")
async def save_workflow(request: WorkflowSaveRequest) -> dict[str, Any]:
    try:
        workflow = workflow_registry.save_workflow(request)
        return workflow.model_dump(mode="json")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/workflows/{workflow_id}/analysis")
async def workflow_analysis(
    workflow_id: str,
    install_dependencies: bool = Query(False),
) -> dict[str, Any]:
    workflow_path = workflow_registry.find(workflow_id)
    if not workflow_path:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    workflow = workflow_registry.summarize(workflow_path)
    object_info = await _optional_comfy_object_info()
    analysis = workflow_registry.analyze_workflow(
        workflow,
        object_info=object_info,
        install_dependencies=install_dependencies,
    )
    return analysis.model_dump(mode="json")


@app.post("/api/workflows/{workflow_id}/install-dependencies")
async def install_workflow_dependencies(
    workflow_id: str,
    restart_comfy: bool = Query(True),
) -> dict[str, Any]:
    workflow_path = workflow_registry.find(workflow_id)
    if not workflow_path:
        raise HTTPException(status_code=404, detail="Workflow not found.")
    workflow = workflow_registry.summarize(workflow_path)
    object_info = await _optional_comfy_object_info()

    install_analysis = workflow_registry.analyze_workflow(
        workflow,
        object_info=object_info,
        install_dependencies=True,
    )

    should_restart_comfy = restart_comfy and install_analysis.dependencies_installed and await comfy.is_running()
    if should_restart_comfy:
        cleanup_embedded_comfy_artifacts()
        await comfy.restart()
        object_info = await _optional_comfy_object_info()
        refreshed = workflow_registry.analyze_workflow(workflow, object_info=object_info)
        refreshed.dependencies_installed = install_analysis.dependencies_installed
        refreshed.dependency_errors = install_analysis.dependency_errors
        return refreshed.model_dump(mode="json")

    return install_analysis.model_dump(mode="json")


@app.get("/api/comfy/object-info")
async def comfy_object_info(start: bool = Query(False)) -> dict[str, Any]:
    try:
        if not start and not await comfy.is_running():
            return {}
        return await comfy.object_info()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/comfy/start")
async def start_comfy() -> dict[str, Any]:
    try:
        cleanup_embedded_comfy_artifacts()
        await comfy.ensure_running()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"status": "running", "url": comfy.base_url}


@app.post("/api/comfy/free")
async def free_comfy_memory(
    unload_models: bool = Query(True),
    free_memory: bool = Query(True),
    force: bool = Query(False),
) -> dict[str, Any]:
    await _guard_runtime_mutation("free ComfyUI memory", force=force)
    try:
        result = await comfy.free_memory(unload_models=unload_models, free_memory=free_memory)
        cleanup_embedded_comfy_artifacts()
        result["memory"] = await _runtime_memory_snapshot()
        return result
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/comfy/stop")
async def stop_comfy(force: bool = Query(False)) -> dict[str, Any]:
    await _guard_runtime_mutation("stop ComfyUI", force=force)
    _cancel_comfy_idle_release()
    try:
        comfy.stop()
        cleanup_embedded_comfy_artifacts()
        return {"status": "stopped", "url": comfy.base_url, "memory": await _runtime_memory_snapshot()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/comfy/restart")
async def restart_comfy(force: bool = Query(False)) -> dict[str, Any]:
    await _guard_runtime_mutation("restart ComfyUI", force=force)
    _cancel_comfy_idle_release()
    try:
        cleanup_embedded_comfy_artifacts()
        await comfy.restart()
        return {"status": "running", "url": comfy.base_url, "memory": await _runtime_memory_snapshot()}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/api/runtime/memory")
async def runtime_memory() -> dict[str, Any]:
    return await _runtime_memory_snapshot()


@app.post("/api/civitai/resolve")
async def civitai_resolve(request: CivitaiResolveRequest) -> dict[str, Any]:
    try:
        return resolve_civitai_asset(settings, request.url, request.token, target_kind=request.target_kind, preset=request.preset)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/civitai/search")
async def civitai_search(request: CivitaiSearchRequest) -> dict[str, Any]:
    try:
        return search_civitai_models(
            query=request.query,
            token=request.token,
            types=request.types,
            base_model=request.base_model,
            sort=request.sort,
            period=request.period,
            nsfw=request.nsfw,
            limit=request.limit,
            cursor=request.cursor,
        )
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/civitai/download")
async def civitai_download(request: CivitaiDownloadRequest) -> dict[str, Any]:
    try:
        result = download_civitai_asset(
            settings,
            url=request.url,
            token=request.token,
            target_kind=request.target_kind,
            preset=request.preset,
            save_preview=request.save_preview,
        )
        ensure_model_tree(settings)
        return result
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _run_civitai_download_job(job_id: str, request: CivitaiDownloadRequest) -> None:
    try:
        _update_download_job(job_id, {"status": "resolving", "progress": 0, "message": "Resolving Civitai asset"})

        def progress(update: dict[str, Any]) -> None:
            _update_download_job(job_id, update)

        result = await asyncio.to_thread(
            download_civitai_asset,
            settings,
            url=request.url,
            token=request.token,
            target_kind=request.target_kind,
            preset=request.preset,
            save_preview=request.save_preview,
            progress_callback=progress,
        )
        ensure_model_tree(settings)
        _update_download_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "message": "Download complete",
                "model_name": result.get("model_name"),
                "file_name": result.get("file_name"),
                "result": result,
            },
        )
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.post("/api/civitai/download/start")
async def civitai_download_start(request: CivitaiDownloadRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    now = datetime.now().isoformat(timespec="seconds")
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued Civitai download",
        "model_name": "",
        "file_name": "",
        "bytes_downloaded": 0,
        "bytes_total": 0,
        "speed_bps": 0,
        "created_at": now,
        "updated_at": now,
    }
    asyncio.create_task(_run_civitai_download_job(job_id, request))
    return download_jobs[job_id]


@app.get("/api/civitai/download/{job_id}")
async def civitai_download_job(job_id: str) -> dict[str, Any]:
    job = download_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Download job not found.")
    return job


@app.get("/api/civitai/downloads")
async def civitai_downloads() -> dict[str, Any]:
    jobs = list(download_jobs.values())[-30:]
    active = [job for job in jobs if job.get("status") in {"queued", "resolving", "downloading", "saving_preview", "downloaded"}]
    return {"active": len(active), "jobs": jobs}


@app.post("/api/import")
async def import_endpoint(request: ImportRequest) -> dict[str, str]:
    try:
        return import_resource(settings, request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


async def _run_generation_core(request: GenerateRequest, job_id: str | None = None) -> GenerateResponse:
    if not settings.runtime.auto_start_comfy and not await comfy.is_running():
        raise HTTPException(status_code=503, detail="ComfyUI runtime is not running.")

    try:
        _cancel_comfy_idle_release()
        runtime_changed = _apply_runtime_options(request.runtime)
        if (runtime_changed or comfy.runtime_changed_since_start()) and await comfy.is_running():
            if job_id:
                _update_generation_job(job_id, {"status": "starting", "progress": 1, "message": "Restarting ComfyUI for selected runtime profile"}, force=True)
            cleanup_embedded_comfy_artifacts()
            await comfy.restart()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 5, "message": "Resolving selected models and template assets"})
        assets = resolve_generation_assets(settings, request)
        _ensure_ltx_default_distilled_loras(request, assets)
        _ensure_wan_4step_loras(request, assets)
        if request.preset.lower() == "ltx":
            missing_ltx_assets: list[str] = []
            if not assets.get("text_encoder"):
                missing_ltx_assets.append("Gemma text encoder")
            if not assets.get("text_projection"):
                missing_ltx_assets.append("LTX 2.3 text projection")
            if not (assets.get("video_vae") or assets.get("vae")):
                missing_ltx_assets.append("LTX 2.3 video VAE")
            if not assets.get("audio_vae"):
                missing_ltx_assets.append("LTX 2.3 audio VAE")
            if not assets.get("latent_upscale"):
                missing_ltx_assets.append("LTX 2.3 latent upscale model")
            if missing_ltx_assets:
                raise ValueError("LTX 2.3 missing required assets: " + ", ".join(missing_ltx_assets) + ".")
        if request.preset.lower() in {"zimageturbo", "zimage"}:
            missing_zimage_assets: list[str] = []
            if not assets.get("primary_model"):
                missing_zimage_assets.append("z_image_turbo_bf16.safetensors")
            if Path(str(assets.get("text_encoder") or "")).name.lower() != "qwen_3_4b.safetensors":
                missing_zimage_assets.append("qwen_3_4b.safetensors")
            if Path(str(assets.get("vae") or "")).name.lower() != "ae.safetensors":
                missing_zimage_assets.append("ae.safetensors")
            if missing_zimage_assets:
                raise ValueError("Z-Image Turbo missing required assets: " + ", ".join(missing_zimage_assets) + ".")
        reference_image_names = _prepare_reference_images(request)
        reference_image_name = reference_image_names[0] if reference_image_names else None
        mask_image_name = _prepare_mask_image(request)
        controlnet_image_name = _prepare_controlnet_image(request)
        if reference_image_name:
            assets["reference_image"] = reference_image_name
        if reference_image_names:
            assets["reference_images"] = reference_image_names
        if mask_image_name:
            assets["mask_image"] = mask_image_name
        if controlnet_image_name:
            assets["controlnet_image"] = controlnet_image_name
        if assets.get("primary_model") and not request.model_name:
            request.model_name = assets["primary_model"]
        workflow_path = workflow_registry.find(request.workflow_id, request.preset)
        if request.preset.lower() == "ltx" and not request.workflow_id:
            workflow_path = None

        if job_id:
            _update_generation_job(job_id, {"status": "starting", "progress": 1, "message": "Starting embedded ComfyUI"}, force=True)
        await comfy.ensure_running()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 3, "message": "Reading Comfy object registry"})
        object_info = await comfy.object_info()

        if request.workflow_override:
            if job_id:
                _update_generation_job(job_id, {"status": "building", "progress": 7, "message": "Using edited visual workflow from active tab"})
            override = request.workflow_override
            fmt = detect_workflow_format(override)
            if fmt == "ui":
                prompt = convert_ui_to_api(override, object_info)
            elif fmt == "api":
                prompt = override
            else:
                raise ValueError("Active workflow tab is not a valid ComfyUI workflow.")
            prompt = patch_workflow(prompt, request, assets=assets)
        elif workflow_path:
            if job_id:
                _update_generation_job(job_id, {"status": "building", "progress": 7, "message": f"Patching workflow {workflow_path.name}"})
            prompt = workflow_registry.load_api_workflow(workflow_path, request, object_info, assets=assets)
        else:
            checkpoint_name = assets.get("primary_model") or Path(request.model_path or request.model_name or "").name
            if not checkpoint_name:
                raise ValueError("No model selected.")
            if job_id:
                _update_generation_job(job_id, {"status": "building", "progress": 7, "message": f"Using simple default workflow for {checkpoint_name}"})
            if request.preset.lower() == "anima":
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not text_encoder_name:
                    raise ValueError("Anima requires a Qwen3-compatible text encoder in models/text_encoders.")
                if not vae_name:
                    raise ValueError("Anima requires a Qwen image VAE in models/vae.")
                prompt = build_basic_anima_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
                )
            elif request.preset.lower() == "ltx":
                text_encoder_name = assets.get("text_encoder")
                if not text_encoder_name:
                    raise ValueError("LTX requires a Gemma text encoder in models/text_encoders.")
                if not assets.get("text_projection"):
                    raise ValueError("LTX 2.3 requires ltx-2.3 text projection in models/text_encoders or models/checkpoints.")
                if not (assets.get("video_vae") or assets.get("vae")):
                    raise ValueError("LTX 2.3 requires the video VAE in models/vae; Automatic cannot fall back to None.")
                if not assets.get("audio_vae"):
                    raise ValueError("LTX 2.3 requires the audio VAE in models/vae, even when audio output is disabled.")
                if not assets.get("latent_upscale"):
                    raise ValueError("LTX 2.3 requires the latent upscale model in models/latent_upscale_models.")
                if checkpoint_name.lower().endswith(".gguf"):
                    raise ValueError("LTX img2vid default requires an LTX checkpoint file. GGUF workflows can still be loaded explicitly.")
                prompt = build_basic_ltx_img2video_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    reference_image_name,
                    text_projection_name=assets.get("text_projection"),
                    audio_vae_name=assets.get("audio_vae"),
                    video_vae_name=assets.get("video_vae") or assets.get("vae"),
                    latent_upscale_name=assets.get("latent_upscale"),
                )
            elif request.preset.lower() == "wan":
                high_model_name = assets.get("wan_high_model")
                low_model_name = assets.get("wan_low_model")
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not high_model_name or not low_model_name:
                    raise ValueError("WAN 2.2 requires high-noise and low-noise Wan models in models/checkpoints, models/unet or models/diffusion_models.")
                if not text_encoder_name:
                    raise ValueError("WAN 2.2 requires a UMT5 text encoder in models/text_encoders.")
                if not vae_name:
                    raise ValueError("WAN 2.2 requires a Wan VAE in models/vae.")
                wan_first_last_node = None
                if len(reference_image_names) > 1:
                    wan_first_last_node = _available_comfy_node(
                        object_info,
                        "WanFirstLastFrameToVideo",
                        "WanFirstLastFrameToVideoFunModel",
                    )
                prompt = build_basic_wan_i2video_workflow(
                    request,
                    high_model_name,
                    low_model_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
                    reference_end_image_name=reference_image_names[1] if len(reference_image_names) > 1 else None,
                    first_last_frame_node=wan_first_last_node,
                )
            elif request.preset.lower() == "qwen":
                checkpoint_name = assets.get("primary_model") or ""
                if not checkpoint_name:
                    raise ValueError(
                        "QWEN txt2img requires a Qwen-Image base model in models/checkpoints, models/unet or models/diffusion_models. "
                        "The installed Qwen-Image-Edit model is used only with img2img/inpaint reference images."
                    )
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not text_encoder_name:
                    raise ValueError("QWEN requires a Qwen image text encoder in models/text_encoders.")
                if not vae_name:
                    raise ValueError("QWEN requires a Qwen image VAE in models/vae.")
                prompt = build_basic_qwen_image_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
                    reference_image_names=reference_image_names,
                    mask_image_name=mask_image_name,
                )
            elif request.preset.lower() in {"zimageturbo", "zimage"}:
                checkpoint_name = assets.get("primary_model") or ""
                if not checkpoint_name:
                    raise ValueError("Z-Image Turbo requires z_image_turbo_bf16.safetensors in models/diffusion_models, models/unet or models/checkpoints.")
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not text_encoder_name:
                    raise ValueError("Z-Image Turbo requires qwen_3_4b.safetensors in models/text_encoders.")
                if not vae_name:
                    raise ValueError("Z-Image Turbo requires ae.safetensors in models/vae.")
                prompt = build_basic_zimage_turbo_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
                    mask_image_name=mask_image_name,
                )
            elif request.preset.lower() == "flux":
                clip_l_name = assets.get("flux_clip_l")
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not clip_l_name:
                    raise ValueError("Flux requires clip_l.safetensors in models/text_encoders.")
                if not text_encoder_name:
                    raise ValueError("Flux requires a T5 text encoder in models/text_encoders.")
                if not vae_name:
                    raise ValueError("Flux requires an AE/Flux VAE in models/vae.")
                prompt = build_basic_flux_workflow(
                    request,
                    checkpoint_name,
                    clip_l_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
                    mask_image_name=mask_image_name,
                )
            else:
                prompt = build_basic_sd_workflow(
                    request,
                    checkpoint_name,
                    reference_image_name=reference_image_name,
                    mask_image_name=mask_image_name,
                    controlnet_name=assets.get("controlnet_model"),
                    controlnet_image_name=assets.get("controlnet_image"),
                    vae_name=assets.get("vae"),
                )

        _materialize_ltx_director_audio(prompt)

        def progress_callback(update: dict[str, Any]) -> None:
            if job_id:
                _update_generation_job(job_id, update)

        generation_started_at = datetime.now().timestamp()
        prompt_id, outputs = await comfy.run_workflow(prompt, progress_callback=progress_callback)
        if not outputs:
            outputs = await _recover_outputs_from_history(prompt_id, generation_started_at)
        _annotate_output_metadata(outputs, request, assets)
        _cleanup_generation_temp()
        response = GenerateResponse(
            job_id=prompt_id,
            prompt_id=prompt_id,
            status="completed",
            message="Generation completed.",
            outputs=outputs,
        )
        await _release_comfy_memory_if_idle()
        _schedule_comfy_idle_release()
        return response
    except Exception as exc:
        _cleanup_generation_temp()
        await _release_comfy_memory_if_idle()
        _schedule_comfy_idle_release()
        if job_id and generation_jobs.get(job_id, {}).get("status") != "cancelled":
            _update_generation_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)}, force=True)
        raise


async def _run_generation_job(job_id: str, request: GenerateRequest) -> None:
    try:
        if generation_lock.locked():
            position = _generation_queue_position(job_id)
            suffix = f" Queue position {position}." if position else ""
            _update_generation_job(
                job_id,
                {
                    "status": "queued",
                    "progress": 0,
                    "queue_position": position,
                    "message": f"Waiting for the active generation to release VRAM.{suffix}",
                },
                force=True,
            )
        async with generation_lock:
            if generation_jobs.get(job_id, {}).get("status") == "cancelled":
                _console_generation(generation_jobs[job_id], force=True)
                return
            _update_generation_job(job_id, {"queue_position": 1, "message": "Generation has the VRAM lock."}, force=True)
            response = await _run_generation_core(request, job_id=job_id)
        if generation_jobs.get(job_id, {}).get("status") == "cancelled":
            _console_generation(generation_jobs[job_id], force=True)
            return
        generation_jobs[job_id].update(
            {
                "status": "completed",
                "progress": 100,
                "message": response.message,
                "prompt_id": response.prompt_id,
                "outputs": [item for item in response.outputs],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _console_generation(generation_jobs[job_id], force=True)
        await _release_comfy_memory_if_idle()
        _schedule_comfy_idle_release()
    except Exception as exc:
        if generation_jobs.get(job_id, {}).get("status") == "cancelled":
            _console_generation(generation_jobs[job_id], force=True)
            return
        generation_jobs[job_id].update(
            {
                "status": "failed",
                "progress": 100,
                "message": str(exc),
                "error": str(exc),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        _console_generation(generation_jobs[job_id], force=True)
        await _release_comfy_memory_if_idle()
        _schedule_comfy_idle_release()


@app.post("/api/generate/start")
async def generate_start(request: GenerateRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    generation_jobs[job_id] = {
        "job_id": job_id,
        "prompt_id": None,
        "status": "queued",
        "progress": 0,
        "message": "Queued generation.",
        "outputs": [],
        "error": None,
        "queue_position": len(_active_generation_jobs()) + 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "preset": request.preset,
        "workflow_id": request.workflow_id,
    }
    _console_generation(generation_jobs[job_id], force=True)
    asyncio.create_task(_run_generation_job(job_id, request))
    return generation_jobs[job_id]


@app.get("/api/generate/{job_id}")
async def generate_status(job_id: str) -> dict[str, Any]:
    job = generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return {key: value for key, value in job.items() if not key.startswith("_")}


@app.post("/api/generate/{job_id}/cancel")
async def cancel_generation(job_id: str) -> dict[str, Any]:
    job = generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    prompt_id = job.get("prompt_id")
    job.update(
        {
            "status": "cancelled",
            "progress": 100,
            "message": "Generation cancelled.",
            "error": "cancelled",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
    )
    _console_generation(job, force=True)
    try:
        if prompt_id:
            await comfy.interrupt(str(prompt_id))
    except Exception:
        pass
    await _release_comfy_memory_if_idle()
    _schedule_comfy_idle_release()
    return {key: value for key, value in job.items() if not key.startswith("_")}


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        async with generation_lock:
            return await _run_generation_core(request)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/gallery")
async def gallery() -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    if not settings.output_dir.exists():
        return items
    for path in sorted(settings.output_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm"}:
            continue
        relative = path.relative_to(settings.output_dir).as_posix()
        url_path = quote(relative, safe="/")
        media_type = "video" if path.suffix.lower() in {".mp4", ".webm"} else "image"
        metadata = _read_output_metadata(path)
        items.append(
            {
                "title": path.name,
                "filename": path.name,
                "path": str(path),
                "image": f"/outputs/{url_path}",
                "thumb": f"/outputs/{url_path}",
                "media_type": media_type,
                "prompt": metadata.get("prompt", ""),
                "negative": metadata.get("negative", ""),
                "model": metadata.get("model", ""),
                "seed": str(metadata.get("seed", "")),
                "steps": str(metadata.get("steps", "")),
                "cfg": str(metadata.get("cfg", "")),
                "sampler": metadata.get("sampler", ""),
                "scheduler": metadata.get("scheduler", ""),
                "preset": metadata.get("preset", ""),
                "activity": metadata.get("activity", ""),
                "width": metadata.get("width", ""),
                "height": metadata.get("height", ""),
                "workflow_id": metadata.get("workflow_id", ""),
                "loras": metadata.get("loras", []),
                "distilled_loras": metadata.get("distilled_loras", []),
                "controlnet": metadata.get("controlnet", {}),
                "vae": metadata.get("vae", ""),
                "text_encoder": metadata.get("text_encoder", ""),
                "video": metadata.get("video", {}),
                "director": metadata.get("director", {}),
                "metadata": metadata,
                "modified": path.stat().st_mtime,
            }
        )
        if len(items) >= 200:
            break
    return items


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Nexus BTA Backend", "status": "ok"}


@app.get("/api")
async def api_root() -> dict[str, str]:
    return {"name": "Nexus BTA API", "status": "ok"}


@app.get("/ui")
async def ui() -> FileResponse:
    return FileResponse(
        settings.project_root / "index.html",
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )


@app.get("/index.html")
async def index_html() -> FileResponse:
    return FileResponse(
        settings.project_root / "index.html",
        headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"},
    )
