from __future__ import annotations

import asyncio
import base64
import importlib.util
import json
import os
import random
import re
import shutil
import subprocess
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .asset_resolver import resolve_generation_assets
from .civitai import download_civitai_asset, resolve_civitai_asset, search_civitai_models
from .comfy_client import ComfyClient, extract_outputs
from .config import load_settings, save_settings
from .dependencies import custom_node_dependency_status, custom_node_requirements, install_custom_node_dependencies
from .importer import import_resource
from .lora_training import (
    build_train_lora_catalog,
    build_train_lora_job,
    public_train_lora_job,
    train_lora_command_text,
    train_lora_job_root,
)
from .scanner import ensure_model_tree, scan_custom_nodes, scan_models
from .schemas import (
    CivitaiDownloadRequest,
    CivitaiResolveRequest,
    CivitaiSearchRequest,
    CustomNodeUpdateRequest,
    DependencyInstallRequest,
    GenerateRequest,
    GenerateResponse,
    ImportRequest,
    PluginInstallRequest,
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
extras_jobs: dict[str, dict[str, Any]] = {}
download_jobs: dict[str, dict[str, Any]] = {}
train_lora_jobs: dict[str, dict[str, Any]] = {}
generation_lock = asyncio.Lock()
comfy_idle_task: asyncio.Task[None] | None = None
_birefnet_cache: dict[str, Any] = {}
_frame_interpolation_cache: dict[str, Any] = {}

ANSI = {
    "reset": "\033[0m",
    "muted": "\033[90m",
    "red": "\033[91m",
    "green": "\033[92m",
    "yellow": "\033[93m",
    "blue": "\033[94m",
    "magenta": "\033[95m",
    "cyan": "\033[96m",
    "white": "\033[97m",
}


def _seconds_label(seconds: float | int | None) -> str:
    try:
        value = max(0, int(float(seconds or 0)))
    except (TypeError, ValueError):
        value = 0
    minutes, secs = divmod(value, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _console_status(status: str) -> str:
    return {
        "queued": "QUEUE",
        "starting": "START",
        "preparing": "PREP",
        "building": "BUILD",
        "running": "RUN",
        "polling": "SYNC",
        "completed": "DONE",
        "failed": "FAIL",
        "cancelled": "CANCEL",
    }.get(status.lower(), status.upper()[:6])


def _console_message(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "queued").lower()
    message = str(job.get("message") or "").strip()
    if status == "queued":
        position = job.get("queue_position")
        return f"queue pos {position}" if position else "waiting for VRAM"
    if status == "starting":
        return "runtime"
    if status == "preparing":
        return "assets"
    if status == "building":
        return "workflow"
    if status == "polling":
        return "syncing ComfyUI"
    if status == "completed":
        return "output ready"
    if status == "cancelled":
        return "cancelled"
    if status == "failed":
        return message[:96] or "failed"
    if message.lower().startswith("queued "):
        return "queued in ComfyUI"
    if "executing node" in message.lower():
        return "executing"
    if "step " in message.lower():
        return "sampling"
    return message[:96] or "active"


def _console_speed(job: dict[str, Any]) -> str:
    parts: list[str] = []
    steps_per_second = job.get("steps_per_second")
    if isinstance(steps_per_second, (int, float)) and steps_per_second > 0:
        parts.append(f"vel {steps_per_second:.2f} step/s")
    elif isinstance(job.get("progress_per_second"), (int, float)) and job.get("progress_per_second") > 0:
        parts.append(f"vel {float(job['progress_per_second']):.2f}%/s")
    eta_seconds = job.get("eta_seconds")
    if isinstance(eta_seconds, (int, float)) and eta_seconds > 0 and int(job.get("progress") or 0) < 100:
        parts.append(f"eta {_seconds_label(eta_seconds)}")
    elapsed_seconds = job.get("elapsed_seconds")
    if isinstance(elapsed_seconds, (int, float)) and elapsed_seconds >= 0:
        parts.append(f"tempo {_seconds_label(elapsed_seconds)}")
    elif job.get("_queued_monotonic"):
        parts.append(f"fila {_seconds_label(time.monotonic() - float(job['_queued_monotonic']))}")
    return " | ".join(parts)


def _console_stage(job: dict[str, Any]) -> str:
    status = str(job.get("status") or "").lower()
    message = str(job.get("message") or "").lower()
    if job.get("current_step") is not None and job.get("total_steps"):
        return f"sample {job['current_step']}/{job['total_steps']}"
    if status == "queued":
        return "fila"
    if status == "starting":
        return "runtime"
    if status == "preparing":
        return "assets"
    if status == "building":
        return "workflow"
    if status == "polling":
        return "sync"
    if "executing node" in message:
        return "node"
    return status or "active"


def _console_generation(job: dict[str, Any], force: bool = False) -> None:
    progress = int(job.get("progress") or 0)
    bucket = progress // 2
    step_key = (job.get("current_step"), job.get("total_steps"), job.get("node"))
    if (
        not force
        and job.get("_last_bucket") == bucket
        and job.get("_last_status") == job.get("status")
        and job.get("_last_step_key") == step_key
    ):
        return
    job["_last_bucket"] = bucket
    job["_last_status"] = job.get("status")
    job["_last_step_key"] = step_key
    filled = max(0, min(24, round(progress / 4.1667)))
    bar = "#" * filled + "-" * (24 - filled)
    raw_status = str(job.get("status") or "queued")
    status = _console_status(raw_status)
    color = ANSI["green"] if status == "DONE" else ANSI["red"] if status in {"FAIL", "CANCEL"} else ANSI["cyan"]
    timestamp = datetime.now().strftime("%H:%M:%S")
    message = _console_message(job)
    speed = _console_speed(job)
    node = str(job.get("node") or "").strip()
    stage = _console_stage(job)
    node_part = f"node {node}" if node else ""
    status_message = "" if job.get("current_step") is not None and message == "sampling" else message
    detail = " | ".join(part for part in [f"etapa {stage}", node_part, speed, status_message] if part)
    brand = f"{ANSI['red']}NEXUS{ANSI['reset']} {ANSI['cyan']}BTA{ANSI['reset']}"
    try:
        print(
            f"{ANSI['muted']}[{timestamp}]{ANSI['reset']} {brand} "
            f"{color}{status:<6}{ANSI['reset']} {color}{bar}{ANSI['reset']} "
            f"{ANSI['white']}{progress:3d}%{ANSI['reset']} {ANSI['muted']}| {detail}{ANSI['reset']}",
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
    if "progress" in update and update.get("progress") is not None:
        next_progress = int(float(update.get("progress") or 0))
        current_progress = int(float(job.get("progress") or 0))
        next_status = str(update.get("status") or job.get("status") or "").lower()
        if next_status not in {"failed", "completed", "cancelled"} and next_progress < current_progress:
            update = {**update, "progress": current_progress}
    job.update({key: value for key, value in update.items() if value is not None})
    if str(job.get("status") or "").lower() in {"running", "polling"} and not job.get("_started_monotonic"):
        job["_started_monotonic"] = time.monotonic()
    if job.get("_started_monotonic") and "elapsed_seconds" not in update:
        job["elapsed_seconds"] = round(time.monotonic() - float(job["_started_monotonic"]), 1)
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _console_generation(job, force=force)


def _public_generation_job(job: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in job.items() if not key.startswith("_")}
    if job.get("_started_monotonic") and "elapsed_seconds" not in public:
        public["elapsed_seconds"] = round(time.monotonic() - float(job["_started_monotonic"]), 1)
    if job.get("_queued_monotonic") and str(job.get("status") or "").lower() == "queued":
        public["queued_seconds"] = round(time.monotonic() - float(job["_queued_monotonic"]), 1)
    return public


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
    explicit_distilled_slots = bool(request.distilled_loras)
    existing = {
        _normalize_lora_key(getattr(item, "name", ""))
        for item in request.distilled_loras
        if _normalize_lora_key(getattr(item, "name", ""))
        and _normalize_lora_key(getattr(item, "name", "")) not in {"none", "automatic", "auto"}
    }
    additions: list[DistilledLoraSelection] = []
    default_strengths = {
        "distilled_lora_1": 0.80,
        "distilled_lora_2": 0.50,
    }
    if not explicit_distilled_slots:
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


def _is_qwen_edit_lightning_lora_name(value: object) -> bool:
    text = str(value or "").lower()
    return "qwen" in text and "edit" in text and "lightning" in text


def _ensure_qwen_edit_lightning_lora(request: GenerateRequest, assets: dict[str, str]) -> None:
    if request.preset.lower() != "qwen" or request.activity != "img2img":
        return
    auto_lightning = (request.video or {}).get("qwen_auto_edit_lora", True)
    if isinstance(auto_lightning, str):
        auto_lightning = auto_lightning.lower() not in {"false", "0", "off", "none", "no"}
    name = assets.get("qwen_edit_lightning_lora")
    if not name:
        request.distilled_loras = [
            item for item in request.distilled_loras if _is_qwen_edit_lightning_lora_name(getattr(item, "name", ""))
        ]
        return
    normalized = _normalize_lora_key(name)
    cleaned: list[DistilledLoraSelection] = []
    has_edit_lightning = False
    for item in request.distilled_loras:
        item_name = getattr(item, "name", "")
        if not _is_qwen_edit_lightning_lora_name(item_name):
            continue
        cleaned.append(item)
        if _normalize_lora_key(item_name) == normalized:
            has_edit_lightning = True
    if auto_lightning is not False and not has_edit_lightning:
        cleaned.insert(0, DistilledLoraSelection(name=name, strength=1.0))
    request.distilled_loras = cleaned[:1]


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


def _prepare_video_value(value: str, prefix: str = "nexus_base_video") -> str:
    value = (value or "").strip()
    if not value:
        raise ValueError("Base video could not be resolved.")
    if value.startswith("data:video/"):
        return _write_input_data_video(value, prefix)
    source: Path | None = None
    if value.startswith("/outputs/") or "/outputs/" in value:
        relative = _output_relative_from_url(value)
        source = (settings.output_dir / relative).resolve()
    else:
        candidate = Path(value)
        if candidate.exists():
            source = candidate.resolve()
    if not source or not source.exists():
        raise ValueError("Base video could not be resolved.")
    suffix = source.suffix.lower() if source.suffix else ".mp4"
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}{suffix}"
    target = settings.input_dir / filename
    shutil.copy2(source, target)
    return filename


def _prepare_base_video(request: GenerateRequest) -> str | None:
    value = (request.img2img.base_video or "").strip()
    if request.activity != "img2img" or not value:
        return None
    return _prepare_video_value(value)


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
        ("clip_vision", "clip_vision"),
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


def _resolve_generation_seed(request: GenerateRequest) -> int:
    if int(request.seed or -1) >= 0:
        return int(request.seed)
    seed = random.randint(0, 2**32 - 1)
    request.seed = seed
    return seed


def _annotate_output_metadata(outputs: list[dict[str, Any]], request: GenerateRequest, assets: dict[str, Any] | None = None) -> None:
    metadata = _generation_metadata(request, assets)
    for output in outputs:
        output["seed"] = request.seed
        output["metadata"] = metadata
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


def _output_slug(value: str, fallback: str = "generation") -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text[:80] or fallback


def _output_node_kind(node: dict[str, Any]) -> str | None:
    class_lower = str(node.get("class_type") or "").lower()
    if class_lower in {"saveimage", "saveimagewithalpha", "rgba_save"}:
        return "image"
    if class_lower in {"savevideo", "vhs_videocombine", "videocombine", "decodeandsavevideo"}:
        return "video"
    if "filename_prefix" not in (node.get("inputs") or {}):
        return None
    if "video" in class_lower or "gif" in class_lower:
        return "video"
    if "image" in class_lower or "rgba" in class_lower:
        return "image"
    return None


def _apply_output_prefixes(prompt: dict[str, Any], request: GenerateRequest) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = "_".join(
        part
        for part in (
            timestamp,
            _output_slug(request.preset, "preset"),
            _output_slug(request.activity, "generation"),
        )
        if part
    )
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        kind = _output_node_kind(node)
        if not kind:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        current = _output_slug(Path(str(inputs.get("filename_prefix") or "")).name, "")
        suffix = f"_{current}" if current and current.lower() not in {"comfyui", "nexus_bta"} else ""
        inputs["filename_prefix"] = f"{kind}/{base}{suffix}"


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
            if path.stat().st_mtime + 30 < start_timestamp:
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
        for _ in range(12):
            try:
                history = await comfy.history(prompt_id)
                outputs = extract_outputs(history.get(prompt_id, {}))
                if outputs:
                    return outputs
            except Exception:
                pass
            await asyncio.sleep(1)
    return _recent_output_files(start_timestamp - 300)


def _public_extras_job(job: dict[str, Any]) -> dict[str, Any]:
    public = {key: value for key, value in job.items() if not key.startswith("_")}
    public["updated_at"] = job.get("updated_at") or datetime.now().isoformat(timespec="seconds")
    return public


def _update_extras_job(job_id: str, update: dict[str, Any], force: bool = False) -> None:
    job = extras_jobs.get(job_id)
    if not job:
        return
    if job.get("status") == "cancelled" and not force:
        return
    job.update({key: value for key, value in update.items() if value is not None})
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")


def _ffmpeg_binary() -> str:
    binary = shutil.which("ffmpeg")
    if binary:
        return binary
    for candidate in (Path(r"C:\ffmpeg\ffmpeg.exe"), Path(r"C:\Users\jpzin\anaconda3\Library\bin\ffmpeg.exe")):
        if candidate.exists():
            return str(candidate)
    raise RuntimeError("FFmpeg was not found in PATH.")


def _ffprobe_binary() -> str | None:
    binary = shutil.which("ffprobe")
    if binary:
        return binary
    for candidate in (Path(r"C:\ffmpeg\ffprobe.exe"), Path(r"C:\Users\jpzin\anaconda3\Library\bin\ffprobe.exe")):
        if candidate.exists():
            return str(candidate)
    return None


def _safe_upload_name(name: str, fallback: str = "source") -> str:
    suffix = Path(name or "").suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name or fallback).stem).strip("._-") or fallback
    return f"{stem[:48]}{suffix}"


async def _save_extras_uploads(files: list[UploadFile]) -> list[Path]:
    upload_root = settings.temp_dir / "extras_uploads" / uuid.uuid4().hex[:12]
    upload_root.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for index, upload in enumerate(files):
        if not upload.filename:
            continue
        suffix = Path(upload.filename).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".mp4", ".mov", ".webm", ".mkv", ".avi"}:
            continue
        target = upload_root / f"{index + 1:04d}_{_safe_upload_name(upload.filename)}"
        with target.open("wb") as handle:
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
        saved.append(target)
    return saved


async def _save_train_lora_uploads(job_id: str, files: list[UploadFile]) -> list[dict[str, Any]]:
    dataset_dir = train_lora_job_root(settings, job_id) / "dataset"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    saved: list[dict[str, Any]] = []
    for index, upload in enumerate(files):
        if not upload.filename:
            continue
        target = dataset_dir / f"{index + 1:04d}_{_safe_upload_name(upload.filename, 'dataset')}"
        with target.open("wb") as handle:
            size = 0
            while True:
                chunk = await upload.read(1024 * 1024)
                if not chunk:
                    break
                size += len(chunk)
                handle.write(chunk)
        saved.append(
            {
                "filename": upload.filename,
                "saved_name": target.name,
                "path": str(target),
                "size": size,
                "content_type": upload.content_type or "",
            }
        )
    return saved


def _update_train_lora_job(job_id: str, update: dict[str, Any]) -> None:
    job = train_lora_jobs.get(job_id)
    if not job:
        return
    if job.get("status") == "cancelled" and update.get("status") != "cancelled":
        return
    job.update({key: value for key, value in update.items() if value is not None})
    job["updated_at"] = datetime.now().isoformat(timespec="seconds")


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


async def _monitor_train_lora_process(job_id: str) -> None:
    job = train_lora_jobs.get(job_id)
    process: subprocess.Popen[Any] | None = job.get("_process") if job else None
    if not job or not process:
        return
    try:
        while process.poll() is None:
            if train_lora_jobs.get(job_id, {}).get("status") == "cancelled":
                return
            await asyncio.sleep(2)
        if train_lora_jobs.get(job_id, {}).get("status") == "cancelled":
            return
        return_code = process.returncode
        if return_code == 0:
            _update_train_lora_job(
                job_id,
                {"status": "completed", "progress": 100, "message": "Train LoRA job completed.", "completed_at": datetime.now().isoformat(timespec="seconds")},
            )
        else:
            _update_train_lora_job(
                job_id,
                {
                    "status": "failed",
                    "progress": 100,
                    "message": f"Train LoRA runner exited with code {return_code}.",
                    "error": f"exit code {return_code}",
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                },
            )
    finally:
        current = train_lora_jobs.get(job_id)
        if current:
            current.pop("_process", None)
            log_handle = current.pop("_log_handle", None)
            try:
                if log_handle:
                    log_handle.close()
            except Exception:
                pass


def _launch_train_lora_job(job_id: str) -> None:
    job = train_lora_jobs[job_id]
    runner = job.get("runner") or {}
    command = [str(part) for part in runner.get("command") or []]
    cwd = str(runner.get("cwd") or "")
    if not command or not cwd:
        _update_train_lora_job(job_id, {"status": "blocked", "message": runner.get("install_hint") or "Train LoRA runner is not available."})
        return
    log_path = Path((job.get("paths") or {}).get("terminal_log") or (Path((job.get("paths") or {}).get("job") or train_lora_job_root(settings, job_id)) / "runner.log"))
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{datetime.now().isoformat(timespec='seconds')}] Launching runner\n")
        handle.write("command=" + " ".join(command) + "\n\n")
    log_handle = log_path.open("ab")
    try:
        process = subprocess.Popen(command, cwd=cwd, stdout=log_handle, stderr=subprocess.STDOUT)
    except Exception as exc:
        log_handle.close()
        _update_train_lora_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})
        return
    job["_process"] = process
    job["_log_handle"] = log_handle
    _update_train_lora_job(
        job_id,
        {
            "status": "running",
            "progress": 25,
            "message": "Train LoRA runner started.",
            "command": train_lora_command_text(job),
            "log_path": str(log_path),
        },
    )
    asyncio.create_task(_monitor_train_lora_process(job_id))


def _pick_local_folder_dialog(title: str) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        root.destroy()
        return str(selected or "")
    except Exception as exc:
        raise RuntimeError(f"Folder picker unavailable: {exc}") from exc


def _pick_local_file_dialog(title: str, filetypes: list[tuple[str, str]]) -> str:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        selected = filedialog.askopenfilename(title=title, filetypes=filetypes)
        root.destroy()
        return str(selected or "")
    except Exception as exc:
        raise RuntimeError(f"File picker unavailable: {exc}") from exc


def _resolve_extras_source_url(value: str) -> Path | None:
    value = (value or "").strip()
    if not value or value.startswith("blob:") or value.startswith("data:"):
        return None
    if value.startswith("/outputs/") or "/outputs/" in value:
        relative = _output_relative_from_url(value)
        source = (settings.output_dir / relative).resolve()
        if source.exists() and source.is_relative_to(settings.output_dir.resolve()):
            return source
    candidate = Path(value)
    if candidate.exists():
        return candidate.resolve()
    return None


def _extras_output(kind: str, suffix: str, stem: str = "extras") -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = settings.output_dir / "extras" / kind
    folder.mkdir(parents=True, exist_ok=True)
    return folder / f"{timestamp}_{_output_slug(stem, 'extras')}_{uuid.uuid4().hex[:6]}{suffix}"


def _output_item(path: Path, kind: str | None = None) -> dict[str, Any]:
    relative = path.resolve().relative_to(settings.output_dir.resolve()).as_posix()
    suffix = path.suffix.lower()
    media_kind = kind or ("video" if suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"} else "image")
    return {
        "kind": media_kind,
        "filename": path.name,
        "subfolder": "" if path.parent == settings.output_dir else path.parent.relative_to(settings.output_dir).as_posix(),
        "type": "output",
        "path": relative,
        "url": f"/outputs/{quote(relative, safe='/')}",
    }


def _write_extras_metadata(outputs: list[dict[str, Any]], plan: dict[str, Any]) -> None:
    for output in outputs:
        relative = str(output.get("path") or "")
        if not relative:
            continue
        path = (settings.output_dir / relative).resolve()
        if not path.exists() or not path.is_relative_to(settings.output_dir.resolve()):
            continue
        metadata = {
            "preset": "Extras",
            "activity": "extras",
            "mode": plan.get("mode") or plan.get("mediaType"),
            "extras": plan,
            "file": path.name,
            "path": relative.replace("\\", "/"),
            "kind": output.get("kind") or path.suffix.lower().lstrip("."),
        }
        try:
            path.with_suffix(path.suffix + ".nexus.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass


def _image_scale_size(width: int, height: int, plan: dict[str, Any]) -> tuple[int, int]:
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    scale = str(upscale.get("scale") or plan.get("scale") or "2x").lower()
    custom_resolution = upscale.get("custom_resolution") or plan.get("custom_resolution")
    if scale == "custom" and isinstance(custom_resolution, dict):
        custom = custom_resolution
        return max(1, int(custom.get("width") or width)), max(1, int(custom.get("height") or height))
    factor = 4 if scale.startswith("4") else 2
    return max(1, width * factor), max(1, height * factor)


def _load_birefnet_model() -> Any:
    import sys

    import torch
    from safetensors.torch import load_file

    model_path = settings.models_dir / "background_removal" / "birefnet.safetensors"
    if not model_path.exists():
        raise FileNotFoundError(f"BiRefNet model not found: {model_path}")

    cache_key = str(model_path.resolve())
    cached = _birefnet_cache.get(cache_key)
    if cached:
        return cached

    layerstyle_path = settings.custom_nodes_dir / "comfyui_layerstyle" / "py"
    birefnet_path = layerstyle_path / "BiRefNet_v2"
    for candidate in (str(layerstyle_path), str(birefnet_path)):
        if candidate not in sys.path:
            sys.path.insert(0, candidate)

    from BiRefNet_v2.models.birefnet import BiRefNet
    from BiRefNet_v2.utils import check_state_dict

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = BiRefNet(bb_pretrained=False)
    state_dict = check_state_dict(load_file(str(model_path), device="cpu"))
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    _birefnet_cache[cache_key] = {"model": model, "device": device, "name": model_path.name}
    return _birefnet_cache[cache_key]


def _remove_bg_image_birefnet(image: Any, threshold: float = 0.45) -> tuple[Any, Any]:
    import torch
    from torchvision import transforms
    from PIL import Image, ImageEnhance, ImageFilter

    loaded = _load_birefnet_model()
    model = loaded["model"]
    device = loaded["device"]
    original = image.convert("RGB")
    inference_size = (1024, 1024)
    transform_image = transforms.Compose(
        [
            transforms.Resize(inference_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    inference_image = transform_image(original).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(inference_image)[-1].sigmoid().cpu()
    pred = preds[0].squeeze().clamp(0, 1)
    mask = transforms.ToPILImage()(pred).resize(original.size, Image.Resampling.BILINEAR).convert("L")
    mask = ImageEnhance.Brightness(mask).enhance(1.05)
    if threshold > 0:
        black = int(255 * max(0.0, min(0.45, threshold * 0.35)))
        white = int(255 * max(0.55, min(1.0, 1.0 - threshold * 0.08)))
        if white > black:
            scale = 255.0 / max(1, white - black)
            mask = mask.point(lambda px: max(0, min(255, int((px - black) * scale))))
    mask = mask.filter(ImageFilter.GaussianBlur(radius=0.35))
    result = original.convert("RGBA")
    result.putalpha(mask)
    return result, mask


def _remove_bg_image_fallback(image: Any, threshold: float = 0.45) -> tuple[Any, Any]:
    from PIL import Image, ImageChops, ImageFilter, ImageStat

    rgba = image.convert("RGBA")
    width, height = rgba.size
    sample = max(1, min(width, height) // 12)
    corners = [
        rgba.crop((0, 0, sample, sample)),
        rgba.crop((width - sample, 0, width, sample)),
        rgba.crop((0, height - sample, sample, height)),
        rgba.crop((width - sample, height - sample, width, height)),
    ]
    stats = [ImageStat.Stat(corner.convert("RGB")).mean for corner in corners]
    bg = tuple(int(sum(values) / len(values)) for values in zip(*stats))
    bg_image = Image.new("RGB", rgba.size, bg)
    diff = ImageChops.difference(rgba.convert("RGB"), bg_image).convert("L")
    cutoff = max(8, min(245, int(255 * max(0.08, min(0.9, threshold)))))
    mask = diff.point(lambda px: 255 if px > cutoff else 0).filter(ImageFilter.GaussianBlur(radius=1.2))
    result = rgba.copy()
    result.putalpha(mask)
    return result, mask


def _remove_bg_image_model(image: Any, threshold: float = 0.45) -> tuple[Any, Any]:
    try:
        return _remove_bg_image_birefnet(image, threshold)
    except Exception:
        return _remove_bg_image_fallback(image, threshold)


def _process_extras_image(source: Path, plan: dict[str, Any], remove_bg: bool = False) -> list[dict[str, Any]]:
    from PIL import Image

    with Image.open(source) as image:
        working = image.convert("RGBA" if (plan.get("preserve_alpha") or remove_bg) else "RGB")
        mask = None
        if remove_bg:
            working, mask = _remove_bg_image_model(working, float(plan.get("remove_background", {}).get("threshold") or plan.get("threshold") or 0.45))
        else:
            upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
            upscale_enabled = bool(upscale.get("enabled") or plan.get("upscaler") or plan.get("scale"))
            if upscale_enabled:
                target = _image_scale_size(working.width, working.height, plan)
                working = working.resize(target, Image.Resampling.LANCZOS)

        outputs: list[dict[str, Any]] = []
        if remove_bg and plan.get("remove_background", {}).get("output") == "mask" and mask is not None:
            output = _extras_output("image", ".png", "remove_bg_mask")
            mask.save(output)
            outputs.append(_output_item(output, "image"))
            return outputs

        export = str(plan.get("export_format") or ("png" if plan.get("preserve_alpha") or remove_bg else "png")).lower()
        if remove_bg:
            export = "png"
        if export in {"jpg", "jpeg"} and working.mode == "RGBA":
            working = working.convert("RGB")
        suffix = ".jpg" if export in {"jpg", "jpeg"} else f".{export if export in {'png', 'webp'} else 'png'}"
        output = _extras_output("image", suffix, "remove_bg" if remove_bg else "upscale")
        working.save(output)
        outputs.append(_output_item(output, "image"))
        if remove_bg and plan.get("remove_background", {}).get("output") == "both" and mask is not None:
            mask_output = _extras_output("image", ".png", "remove_bg_mask")
            mask.save(mask_output)
            outputs.append(_output_item(mask_output, "image"))
        return outputs


def _ffprobe_fps(source: Path) -> float:
    ffprobe = _ffprobe_binary()
    if not ffprobe:
        return 30.0
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=avg_frame_rate",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        raw = result.stdout.strip()
        if "/" in raw:
            num, den = raw.split("/", 1)
            value = float(num) / max(1.0, float(den))
        else:
            value = float(raw)
        return value if value > 0 else 30.0
    except Exception:
        return 30.0


def _prepare_sequence_input(files: list[Path]) -> tuple[Path, str]:
    from PIL import Image

    sequence_dir = settings.temp_dir / "extras_sequences" / uuid.uuid4().hex[:12]
    sequence_dir.mkdir(parents=True, exist_ok=True)
    for index, source in enumerate(files, start=1):
        with Image.open(source) as image:
            image.convert("RGBA").save(sequence_dir / f"frame_{index:06d}.png")
    return sequence_dir, str(sequence_dir / "frame_%06d.png")


def _load_frame_interpolation_model(model_name: str | None = None) -> Any:
    import sys

    comfy_root = settings.comfy_root
    comfy_root_text = str(comfy_root)
    if comfy_root_text not in sys.path:
        sys.path.insert(0, comfy_root_text)

    import folder_paths
    from comfy_extras.nodes_frame_interpolation import FrameInterpolationModelLoader

    model_folder = settings.models_dir / "frame_interpolation"
    folder_paths.add_model_folder_path("frame_interpolation", str(model_folder))
    selected = Path(str(model_name or "rife_v4.26.safetensors")).name
    if selected.lower() in {"", "automatic", "auto", "off", "none", "no interpolation model detected"}:
        selected = "rife_v4.26.safetensors"
    cache_key = selected.lower()
    if cache_key not in _frame_interpolation_cache:
        _frame_interpolation_cache[cache_key] = FrameInterpolationModelLoader.execute(selected)[0]
    return _frame_interpolation_cache[cache_key]


def _pil_to_comfy_tensor(path: Path) -> Any:
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        array = np.asarray(rgb).astype("float32") / 255.0
    return torch.from_numpy(array)


def _save_comfy_tensor_image(frame: Any, path: Path) -> None:
    import numpy as np
    from PIL import Image

    array = (frame.detach().cpu().clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(array, "RGB").save(path)


def _rife_interpolate_frame_paths(source_frames: list[Path], output_dir: Path, source_fps: float, target_fps: float, model_name: str | None = None) -> tuple[list[Path], float]:
    import math

    import torch
    from PIL import Image

    if len(source_frames) < 2 or target_fps <= source_fps + 0.01:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for index, frame in enumerate(source_frames, start=1):
            target = output_dir / f"frame_{index:06d}.png"
            shutil.copy2(frame, target)
            outputs.append(target)
        return outputs, source_fps

    comfy_root_text = str(settings.comfy_root)
    import sys
    if comfy_root_text not in sys.path:
        sys.path.insert(0, comfy_root_text)
    import comfy.utils
    from comfy import model_management
    from comfy.ldm.common_dit import pad_to_patch_size

    interp_model = _load_frame_interpolation_model(model_name)
    with Image.open(source_frames[0]) as first_image:
        width, height = first_image.convert("RGB").size

    num_frames = len(source_frames)
    target_count = max(num_frames, int(round((num_frames / max(1e-6, source_fps)) * target_fps)))

    device = interp_model.load_device
    dtype = interp_model.model_dtype()
    inference_model = interp_model.model
    sample_shape = (1, height, width, 3)
    activation_mem = inference_model.memory_used_forward(sample_shape, dtype)
    model_management.load_models_gpu([interp_model], memory_required=activation_mem)
    align = getattr(inference_model, "pad_align", 1)
    out_dtype = model_management.intermediate_dtype()
    offload_device = model_management.intermediate_device()
    output_dir.mkdir(parents=True, exist_ok=True)

    def prepare_frame(idx: int) -> Any:
        frame = _pil_to_comfy_tensor(source_frames[idx]).unsqueeze(0).movedim(-1, 1).to(dtype=dtype, device=device)
        if align > 1:
            frame = pad_to_patch_size(frame, (align, align), padding_mode="reflect")
        return frame

    def output_path(output_index: int) -> Path:
        return output_dir / f"frame_{output_index + 1:06d}.png"

    output_slots: list[tuple[int, int | None, float]] = []
    grouped: dict[int, list[tuple[int, float]]] = {}
    for output_index in range(target_count):
        source_position = output_index * source_fps / target_fps
        pair_index = int(math.floor(source_position))
        fraction = source_position - pair_index
        if pair_index >= num_frames - 1:
            output_slots.append((output_index, num_frames - 1, 0.0))
        elif fraction <= 1e-4:
            output_slots.append((output_index, pair_index, 0.0))
        elif fraction >= 1.0 - 1e-4:
            output_slots.append((output_index, pair_index + 1, 0.0))
        else:
            output_slots.append((output_index, None, fraction))
            grouped.setdefault(pair_index, []).append((output_index, fraction))

    output_paths = [output_path(index) for index in range(target_count)]
    for output_index, original_index, _ in output_slots:
        if original_index is not None:
            _save_comfy_tensor_image(_pil_to_comfy_tensor(source_frames[original_index]), output_path(output_index))

    pbar = comfy.utils.ProgressBar(sum(len(items) for items in grouped.values()))
    multi_fn = getattr(inference_model, "forward_multi_timestep", None)
    batch_limit = 1
    prev_frame = None
    feat_cache: dict[str, Any] = {}

    for pair_index in range(num_frames - 1):
        items = grouped.get(pair_index)
        if not items:
            prev_frame = prepare_frame(pair_index + 1)
            feat_cache.pop("img0", None)
            feat_cache.pop("img1", None)
            feat_cache.pop("next", None)
            continue

        img0_single = prev_frame if prev_frame is not None else prepare_frame(pair_index)
        img1_single = prepare_frame(pair_index + 1)
        prev_frame = img1_single
        feat_cache["img0"] = feat_cache.pop("next") if "next" in feat_cache else inference_model.extract_features(img0_single)
        feat_cache["img1"] = inference_model.extract_features(img1_single)
        feat_cache["next"] = feat_cache["img1"]

        start = 0
        while start < len(items):
            chunk = items[start:start + batch_limit]
            timestep_values = [fraction for _, fraction in chunk]
            try:
                with torch.inference_mode():
                    if multi_fn is not None:
                        mids = multi_fn(img0_single, img1_single, timestep_values, cache=feat_cache)
                    else:
                        sample = img0_single
                        p_height, p_width = sample.shape[2], sample.shape[3]
                        ts = torch.tensor(timestep_values, device=device, dtype=dtype).reshape(len(chunk), 1, 1, 1)
                        ts = ts.expand(-1, 1, p_height, p_width)
                        mids = inference_model(
                            img0_single.expand(len(chunk), -1, -1, -1),
                            img1_single.expand(len(chunk), -1, -1, -1),
                            timestep=ts,
                            cache=feat_cache,
                        )
                for mid, (output_index, _) in zip(mids, chunk):
                    frame = mid[:, :height, :width].movedim(0, -1).to(dtype=out_dtype, device=offload_device)
                    _save_comfy_tensor_image(frame, output_path(output_index))
                pbar.update(len(chunk))
                start += len(chunk)
            except model_management.OOM_EXCEPTION:
                if batch_limit <= 1:
                    raise
                batch_limit = max(1, batch_limit // 2)
                model_management.soft_empty_cache()
        model_management.soft_empty_cache()

    return output_paths, target_fps


def _extract_video_frames_for_extras(source_files: list[Path], source_fps: float, target_dir: Path) -> tuple[list[Path], float]:
    image_files = [path for path in source_files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    video_files = [path for path in source_files if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}]
    target_dir.mkdir(parents=True, exist_ok=True)
    if len(image_files) > 1:
        output_paths: list[Path] = []
        for index, frame in enumerate(sorted(image_files, key=lambda item: item.name.lower()), start=1):
            target = target_dir / f"frame_{index:06d}.png"
            shutil.copy2(frame, target)
            output_paths.append(target)
        return output_paths, source_fps or 30.0
    if image_files:
        target = target_dir / "frame_000001.png"
        shutil.copy2(image_files[0], target)
        return [target], source_fps or 1.0
    if not video_files:
        raise ValueError("No video or image sequence source was provided.")
    source = video_files[0]
    detected_fps = source_fps or _ffprobe_fps(source)
    pattern = target_dir / "frame_%06d.png"
    _run_ffmpeg([_ffmpeg_binary(), "-y", "-i", str(source), str(pattern)])
    frames = sorted(target_dir.glob("frame_*.png"))
    if not frames:
        raise RuntimeError("FFmpeg did not extract any frames for interpolation.")
    return frames, detected_fps


def _run_ffmpeg(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = (result.stderr or result.stdout or "FFmpeg failed.").strip()
        raise RuntimeError(message[-1800:])


def _video_encoder_args(encoder: str, output: Path) -> list[str]:
    encoder = (encoder or "mp4_h264").lower()
    if encoder == "webm_vp9":
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "30", "-b:v", "0", str(output)]
    if encoder == "mov_prores_4444_alpha":
        return ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", str(output)]
    if encoder == "mov_prores_422":
        return ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", str(output)]
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "medium", str(output)]


def _encode_extras_frame_sequence(frames: list[Path], fps: float, encoder: str, stem: str) -> list[dict[str, Any]]:
    if not frames:
        return []
    source_dir = frames[0].parent
    pattern = source_dir / "frame_%06d.png"
    encoder = str(encoder or "mp4_h264")
    if encoder.startswith("image_sequence"):
        outputs: list[dict[str, Any]] = []
        for frame in sorted(frames)[:12]:
            if frame.resolve().is_relative_to(settings.output_dir.resolve()):
                outputs.append(_output_item(frame, "image"))
        return outputs
    suffix = ".webm" if encoder == "webm_vp9" else ".mov" if encoder.startswith("mov_") else ".mp4"
    output = _extras_output("video", suffix, stem)
    command = [_ffmpeg_binary(), "-y", "-framerate", str(max(1.0, fps)), "-i", str(pattern), *_video_encoder_args(encoder, output)]
    _run_ffmpeg(command)
    return [_output_item(output, "video")]


def _process_extras_remove_bg_video(source_files: list[Path], plan: dict[str, Any]) -> list[dict[str, Any]]:
    source_fps = float(plan.get("source_fps") or 0)
    work_dir = settings.temp_dir / "extras_remove_bg_frames" / uuid.uuid4().hex[:12]
    input_dir = work_dir / "input"
    output_dir = settings.output_dir / "extras" / "video" / datetime.now().strftime("%Y%m%d_%H%M%S")
    source_frames, detected_fps = _extract_video_frames_for_extras(source_files, source_fps, input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    threshold = float(plan.get("remove_background", {}).get("threshold") or 0.45)
    from PIL import Image

    output_frames: list[Path] = []
    for index, frame_path in enumerate(source_frames, start=1):
        with Image.open(frame_path) as frame:
            rgba, _ = _remove_bg_image_model(frame, threshold)
        target = output_dir / f"frame_{index:06d}.png"
        rgba.save(target)
        output_frames.append(target)
    return _encode_extras_frame_sequence(output_frames, detected_fps, str(plan.get("encoder") or "image_sequence_png_alpha"), "remove_bg_video")


def _process_extras_video(source_files: list[Path], plan: dict[str, Any], remove_bg: bool = False) -> list[dict[str, Any]]:
    if remove_bg:
        return _process_extras_remove_bg_video(source_files, plan)

    ffmpeg = _ffmpeg_binary()
    image_files = [path for path in source_files if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".bmp"}]
    video_files = [path for path in source_files if path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}]
    source_fps = float(plan.get("source_fps") or 0)
    interpolate = plan.get("interpolate") or {}
    target_fps = max(1.0, min(240.0, float(interpolate.get("fps") or plan.get("target_fps") or source_fps or 30.0)))

    input_args: list[str]
    active_fps = source_fps
    intermediate_dir: Path | None = None
    if interpolate.get("enabled"):
        frame_root = settings.temp_dir / "extras_interpolation" / uuid.uuid4().hex[:12]
        source_frames, detected_fps = _extract_video_frames_for_extras(source_files, source_fps, frame_root / "source")
        active_fps = detected_fps
        interpolated_frames, active_fps = _rife_interpolate_frame_paths(
            source_frames,
            frame_root / "rife",
            detected_fps,
            target_fps,
            str(interpolate.get("model") or "rife_v4.26.safetensors"),
        )
        intermediate_dir = interpolated_frames[0].parent
        input_args = ["-framerate", str(active_fps), "-i", str(intermediate_dir / "frame_%06d.png")]
    elif len(image_files) > 1:
        _, sequence_pattern = _prepare_sequence_input(sorted(image_files, key=lambda item: item.name.lower()))
        active_fps = source_fps or 30.0
        input_args = ["-framerate", str(active_fps), "-i", sequence_pattern]
    elif video_files:
        source = video_files[0]
        active_fps = source_fps or _ffprobe_fps(source)
        input_args = ["-i", str(source)]
    elif image_files:
        active_fps = source_fps or 1.0
        input_args = ["-loop", "1", "-framerate", str(active_fps), "-t", "1", "-i", str(image_files[0])]
    else:
        raise ValueError("No video or image sequence source was provided.")

    filters: list[str] = []
    upscale = plan.get("upscale") or {}
    if upscale.get("enabled"):
        factor = 4 if str(upscale.get("scale") or "2x").startswith("4") else 2
        filters.append(f"scale=iw*{factor}:ih*{factor}:flags=lanczos")
    denoise = plan.get("denoise") or {}
    if denoise.get("enabled"):
        filters.append("hqdn3d=1.5:1.5:6:6")
    if plan.get("preserve_alpha"):
        filters.append("format=rgba")

    encoder = str(plan.get("encoder") or ("image_sequence_png_alpha" if remove_bg else "mp4_h264"))
    outputs: list[dict[str, Any]] = []
    command = [ffmpeg, "-y", *input_args]
    if filters:
        command.extend(["-vf", ",".join(filters)])

    if encoder.startswith("image_sequence"):
        folder = settings.output_dir / "extras" / "video" / datetime.now().strftime("%Y%m%d_%H%M%S")
        folder.mkdir(parents=True, exist_ok=True)
        pattern = folder / "frame_%06d.png"
        _run_ffmpeg([*command, str(pattern)])
        for frame in sorted(folder.glob("frame_*.png"))[:12]:
            outputs.append(_output_item(frame, "image"))
        return outputs

    suffix = ".webm" if encoder == "webm_vp9" else ".mov" if encoder.startswith("mov_") else ".mp4"
    output = _extras_output("video", suffix, "remove_bg_video" if remove_bg else "video")
    _run_ffmpeg([*command, *_video_encoder_args(encoder, output)])
    outputs.append(_output_item(output, "video"))
    return outputs


async def _run_extras_job(job_id: str, source_files: list[Path], plan: dict[str, Any]) -> None:
    try:
        _update_extras_job(job_id, {"status": "running", "progress": 8, "message": "Preparing Extras source."})
        mode = str(plan.get("mode") or "").lower()
        remove_bg = mode == "remove_bg"
        media_type = str(plan.get("mediaType") or "").lower()
        if not source_files and plan.get("source_url"):
            source = _resolve_extras_source_url(str(plan.get("source_url") or ""))
            if source:
                source_files = [source]
        if not source_files:
            raise ValueError("No source files were received by Extras.")

        _update_extras_job(job_id, {"progress": 18, "message": "Running Extras pipeline."})
        video_exts = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
        is_video = media_type in {"video", "image_sequence"} or any(path.suffix.lower() in video_exts for path in source_files) or len(source_files) > 1
        if is_video:
            outputs = await asyncio.to_thread(_process_extras_video, source_files, plan, remove_bg)
        else:
            outputs = await asyncio.to_thread(_process_extras_image, source_files[0], plan, remove_bg)
        _write_extras_metadata(outputs, plan)
        _update_extras_job(
            job_id,
            {
                "status": "completed",
                "progress": 100,
                "message": "Extras completed.",
                "outputs": outputs,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
            force=True,
        )
    except Exception as exc:
        _update_extras_job(
            job_id,
            {
                "status": "failed",
                "progress": 100,
                "message": str(exc),
                "error": str(exc),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
            force=True,
        )


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


def _json_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _coerce_number(value: Any, default: Any = None) -> Any:
    if value in (None, ""):
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return int(parsed) if parsed.is_integer() else parsed


def _parse_a1111_parameters(text: str) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if not text:
        return result
    prompt_text = text
    settings_line = ""
    if "\nSteps:" in text:
        prompt_text, settings_line = text.rsplit("\n", 1)
    if "Negative prompt:" in prompt_text:
        positive, negative = prompt_text.split("Negative prompt:", 1)
        result["prompt"] = positive.strip()
        result["negative_prompt"] = negative.strip()
    else:
        result["prompt"] = prompt_text.strip()
    key_map = {
        "Steps": "steps",
        "Sampler": "sampler",
        "Schedule type": "scheduler",
        "CFG scale": "cfg",
        "Seed": "seed",
        "Model": "model",
        "Size": "size",
    }
    for match in re.finditer(r"([^,:\n]+):\s*([^,]+)", settings_line):
        key = key_map.get(match.group(1).strip())
        if not key:
            continue
        value = match.group(2).strip()
        result[key] = _coerce_number(value, value)
    size = str(result.pop("size", "") or "")
    size_match = re.match(r"(\d+)\s*x\s*(\d+)", size)
    if size_match:
        result["width"] = int(size_match.group(1))
        result["height"] = int(size_match.group(2))
    return result


def _extract_comfy_prompt_settings(prompt: dict[str, Any]) -> dict[str, Any]:
    settings_out: dict[str, Any] = {}
    text_values: list[str] = []
    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "").lower()
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        if "sampler" in class_type:
            for source, target in (
                ("seed", "seed"),
                ("noise_seed", "seed"),
                ("steps", "steps"),
                ("cfg", "cfg"),
                ("sampler_name", "sampler"),
                ("scheduler", "scheduler"),
            ):
                if inputs.get(source) not in (None, ""):
                    settings_out[target] = inputs.get(source)
        if "checkpointloader" in class_type and inputs.get("ckpt_name"):
            settings_out["model"] = inputs.get("ckpt_name")
        if "unetloader" in class_type and inputs.get("unet_name"):
            settings_out["model"] = inputs.get("unet_name")
        if "emptylatent" in class_type or "emptyhunyuanlatent" in class_type:
            if inputs.get("width"):
                settings_out["width"] = inputs.get("width")
            if inputs.get("height"):
                settings_out["height"] = inputs.get("height")
        text = inputs.get("text")
        if isinstance(text, str) and text.strip():
            text_values.append(text.strip())
    if text_values:
        settings_out.setdefault("prompt", text_values[0])
    if len(text_values) > 1:
        settings_out.setdefault("negative_prompt", text_values[1])
    return settings_out


def _normalize_png_info(raw_chunks: dict[str, str], width: int, height: int) -> dict[str, Any]:
    nexus_meta = _json_dict(raw_chunks.get("nexus_bta"))
    prompt_json = _json_dict(raw_chunks.get("prompt"))
    workflow_json = _json_dict(raw_chunks.get("workflow"))
    a1111_meta = _parse_a1111_parameters(raw_chunks.get("parameters", ""))
    comfy_settings = _extract_comfy_prompt_settings(prompt_json)
    summary = {
        **a1111_meta,
        **comfy_settings,
        **nexus_meta,
    }
    summary.setdefault("width", width)
    summary.setdefault("height", height)
    source = "none"
    if nexus_meta:
        source = "nexus_bta"
    elif prompt_json or workflow_json:
        source = "comfyui"
    elif a1111_meta:
        source = "parameters"
    return {
        "source": source,
        "summary": summary,
        "metadata": nexus_meta,
        "prompt_json": prompt_json,
        "workflow_json": workflow_json,
        "parameters": raw_chunks.get("parameters", ""),
        "raw_text_chunks": raw_chunks,
    }


def _run_git(args: list[str], cwd: Path, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, timeout=timeout)


def _resolve_custom_node_path(node_name: str) -> Path:
    requested = str(node_name or "").strip()
    if not requested or any(char in requested for char in ("/", "\\")):
        raise HTTPException(status_code=400, detail="Invalid custom node name.")
    custom_root = settings.custom_nodes_dir.resolve()
    for node in scan_custom_nodes(settings):
        if node.name.lower() == requested.lower():
            path = Path(node.path).resolve()
            if path.exists() and path.is_relative_to(custom_root):
                return path
    raise HTTPException(status_code=404, detail="Custom node not found.")


def _custom_node_git_versions(path: Path) -> dict[str, Any]:
    if not (path / ".git").exists():
        return {"git": False, "current": "", "tags": [], "branches": [], "default_version": ""}
    try:
        _run_git(["fetch", "--all", "--tags", "--prune"], path, timeout=90)
    except Exception:
        pass
    current = _run_git(["rev-parse", "--abbrev-ref", "HEAD"], path)
    short = _run_git(["rev-parse", "--short", "HEAD"], path)
    tags = _run_git(["tag", "--sort=-creatordate"], path)
    branches = _run_git(["branch", "-r", "--format=%(refname:short)"], path)
    branch_items = [
        line.strip().replace("origin/", "", 1)
        for line in branches.stdout.splitlines()
        if line.strip() and "HEAD" not in line
    ]
    return {
        "git": True,
        "current": f"{current.stdout.strip()}@{short.stdout.strip()}".strip("@"),
        "tags": [line.strip() for line in tags.stdout.splitlines() if line.strip()][:80],
        "branches": sorted(set(branch_items))[:80],
        "default_version": "",
    }


def _update_custom_node(path: Path, version: str = "") -> dict[str, Any]:
    if not (path / ".git").exists():
        raise HTTPException(status_code=400, detail=f"{path.name} is not a git extension.")
    fetch = _run_git(["fetch", "--all", "--tags", "--prune"], path)
    if fetch.returncode != 0:
        raise HTTPException(status_code=500, detail=(fetch.stderr or fetch.stdout or "git fetch failed").strip()[-1200:])
    target = str(version or "").strip()
    if target:
        checkout = _run_git(["checkout", target], path)
        if checkout.returncode != 0:
            raise HTTPException(status_code=500, detail=(checkout.stderr or checkout.stdout or "git checkout failed").strip()[-1200:])
    pull = _run_git(["pull", "--ff-only"], path)
    if pull.returncode != 0 and not target:
        raise HTTPException(status_code=500, detail=(pull.stderr or pull.stdout or "git pull failed").strip()[-1200:])
    versions = _custom_node_git_versions(path)
    return {"name": path.name, "updated": True, "version": versions.get("current", ""), "output": (pull.stdout or checkout.stdout if target else pull.stdout)[-1200:]}


def _plugin_repo_name(url: str) -> str:
    parsed = urlparse(url.strip())
    name = Path(parsed.path.rstrip("/")).name or "plugin"
    if name.endswith(".git"):
        name = name[:-4]
    name = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip(".-_")
    if not name:
        raise HTTPException(status_code=400, detail="Invalid GitHub plugin URL.")
    return name[:80]


def _find_video2video_workflow(preset: str) -> Path | None:
    preset_lower = preset.lower()
    workflows = workflow_registry.list_workflows()
    for workflow in workflows:
        classes = {item.lower() for item in workflow.class_types}
        name = f"{workflow.id} {workflow.name}".lower()
        has_video_loader = any(item in classes for item in {"loadvideo", "vhs_loadvideo", "loadvideoui"})
        if preset_lower == "wan" and has_video_loader and (
            "wanvideoanimateembeds" in classes or "wan animate" in name or "wan2-2-animate" in name
        ):
            return Path(workflow.path)
        if preset_lower == "ltx" and has_video_loader and (
            "ltxaddvideoicloraguide" in classes or "ic-lora" in name or "union-control" in name
        ):
            return Path(workflow.path)
    return None


def _cleanup_generation_temp() -> None:
    for pattern in ("nexus_reference_*", "nexus_base_video_*", "nexus_mask_*", "nexus_controlnet_*", "nexus_director_audio_*"):
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
    if text in {"gpu", "gpuonly", "onlygpu", "cudaonly"}:
        return "gpu_only"
    if text in {"shared", "vramshared", "sharedvram", "dynamic", "default", "auto", "low", "lowvram", "med", "medium", "medvram", "normal", "balanced", "balance", "high", "highvram"}:
        return "shared"
    return "shared"


def _canonical_gpu_memory_gb(value: float | int | str | None) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if parsed <= 0:
        return None
    return round(max(1.0, min(parsed, 192.0)), 2)


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
    next_gpu_memory_gb = _canonical_gpu_memory_gb(options.gpu_memory_gb)
    next_attention = _canonical_attention_backend(options.attention_backend)
    next_precision = _canonical_precision(options.precision)
    next_disable_xformers = bool(options.disable_xformers)
    changed = (
        _canonical_vram_policy(settings.runtime.vram_policy) != next_vram
        or _canonical_gpu_memory_gb(settings.runtime.gpu_memory_gb) != next_gpu_memory_gb
        or _canonical_attention_backend(settings.runtime.attention_backend) != next_attention
        or _canonical_precision(settings.runtime.precision) != next_precision
        or bool(settings.runtime.disable_xformers) != next_disable_xformers
    )
    settings.runtime.vram_policy = next_vram
    settings.runtime.gpu_memory_gb = next_gpu_memory_gb
    settings.runtime.attention_backend = next_attention
    settings.runtime.precision = next_precision
    settings.runtime.disable_xformers = next_disable_xformers
    settings.runtime.enable_sage_attention = next_attention == "sage"
    settings.runtime.enable_flash_attention = next_attention == "flash"
    if changed:
        save_settings(settings)
    return changed


def _prepare_runtime_for_generation(request: GenerateRequest) -> None:
    if request.preset.lower() == "qwen" and request.activity == "img2img":
        request.runtime.attention_backend = "pytorch"
        request.runtime.disable_xformers = True


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


def _runtime_attention_capabilities() -> dict[str, Any]:
    def module_status(module_name: str) -> dict[str, Any]:
        available = importlib.util.find_spec(module_name) is not None
        result: dict[str, Any] = {"available": available, "version": "", "error": ""}
        if not available:
            return result
        try:
            module = __import__(module_name)
            result["version"] = str(getattr(module, "__version__", "") or "")
        except Exception as exc:
            result["available"] = False
            result["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"
        return result

    torch_info: dict[str, Any] = {
        "available": False,
        "version": "",
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_name": "",
        "cuda_total_vram_bytes": 0,
    }
    try:
        import torch

        torch_info["available"] = True
        torch_info["version"] = str(getattr(torch, "__version__", "") or "")
        cuda_available = bool(torch.cuda.is_available())
        torch_info["cuda_available"] = cuda_available
        if cuda_available:
            torch_info["cuda_device_count"] = int(torch.cuda.device_count())
            torch_info["cuda_name"] = str(torch.cuda.get_device_name(0))
            torch_info["cuda_total_vram_bytes"] = int(torch.cuda.get_device_properties(0).total_memory)
    except Exception as exc:
        torch_info["error"] = f"{type(exc).__name__}: {str(exc)[:180]}"

    modules = {
        "xformers": module_status("xformers"),
        "sageattention": module_status("sageattention"),
        "flash_attn": module_status("flash_attn"),
    }
    vram_gb = float(torch_info.get("cuda_total_vram_bytes") or 0) / (1024**3)
    recommended_attention = "auto"
    if modules["xformers"]["available"]:
        recommended_attention = "auto"
    elif modules["sageattention"]["available"]:
        recommended_attention = "sage"
    elif modules["flash_attn"]["available"]:
        recommended_attention = "flash"
    elif not modules["xformers"]["available"]:
        recommended_attention = "pytorch"

    recommended_vram = "shared"
    recommended_gpu_memory_gb = None
    if torch_info.get("cuda_available") and vram_gb > 0:
        recommended_gpu_memory_gb = round(max(1.0, min(vram_gb - 1.5, vram_gb * 0.85)), 1)

    return {
        "torch": torch_info,
        "attention_modules": modules,
        "recommended": {
            "attention_backend": recommended_attention,
            "disable_xformers": not modules["xformers"]["available"],
            "vram_policy": recommended_vram,
            "gpu_memory_gb": recommended_gpu_memory_gb,
            "precision": "auto",
        },
    }


app = FastAPI(title="Nexus BTA Backend", version="0.1.1")
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
react_dist = settings.project_root / "frontend" / "dist"
react_static_dir = react_dist / "static"
if react_static_dir.exists():
    app.mount("/app/static", StaticFiles(directory=react_static_dir), name="react-static")


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
async def custom_node_dependencies() -> dict[str, Any]:
    return custom_node_dependency_status(settings)


@app.post("/api/custom-nodes/install-dependencies")
async def install_dependencies(request: DependencyInstallRequest) -> dict[str, Any]:
    installed, errors = install_custom_node_dependencies(
        settings,
        node_names=request.node_names,
        all_enabled=request.all_enabled,
    )
    return {"installed": installed, "errors": errors}


@app.post("/api/png-info")
async def png_info(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file was uploaded.")
    content = await file.read()
    try:
        from PIL import Image

        with Image.open(BytesIO(content)) as image:
            raw_chunks = {str(key): str(value) for key, value in (getattr(image, "text", {}) or {}).items()}
            width, height = image.size
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image metadata: {exc}") from exc
    return _normalize_png_info(raw_chunks, width, height)


@app.get("/api/custom-nodes/{node_name}/versions")
async def custom_node_versions(node_name: str) -> dict[str, Any]:
    path = _resolve_custom_node_path(unquote(node_name))
    return {"name": path.name, **_custom_node_git_versions(path)}


@app.post("/api/custom-nodes/{node_name}/update")
async def update_custom_node(node_name: str, request: CustomNodeUpdateRequest) -> dict[str, Any]:
    path = _resolve_custom_node_path(unquote(node_name))
    return _update_custom_node(path, request.version)


@app.post("/api/custom-nodes/update-all")
async def update_all_custom_nodes() -> dict[str, Any]:
    updated: list[dict[str, Any]] = []
    errors: dict[str, str] = {}
    for node in scan_custom_nodes(settings):
        path = Path(node.path)
        if not (path / ".git").exists():
            continue
        try:
            updated.append(_update_custom_node(path, ""))
        except Exception as exc:
            errors[node.name] = str(getattr(exc, "detail", exc))[-1200:]
    return {"updated": updated, "errors": errors}


@app.get("/api/plugins")
async def plugins() -> list[dict[str, Any]]:
    return [
        {
            **node.model_dump(mode="json"),
            "plugin": True,
            "git": (Path(node.path) / ".git").exists(),
        }
        for node in scan_custom_nodes(settings)
    ]


@app.post("/api/plugins/install")
async def install_plugin(request: PluginInstallRequest) -> dict[str, Any]:
    url = request.url.strip()
    if not re.match(r"^https://github\.com/[^/\s]+/[^/\s]+/?(?:\.git)?$", url, re.I):
        raise HTTPException(status_code=400, detail="Use a valid GitHub repository URL.")
    name = _plugin_repo_name(url)
    root = settings.custom_nodes_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = (root / name).resolve()
    if not target.is_relative_to(root):
        raise HTTPException(status_code=400, detail="Invalid plugin target.")
    if target.exists():
        raise HTTPException(status_code=409, detail=f"{name} is already installed.")
    clone = _run_git(["clone", url, str(target)], root, timeout=600)
    if clone.returncode != 0:
        raise HTTPException(status_code=500, detail=(clone.stderr or clone.stdout or "git clone failed").strip()[-1200:])
    installed: list[str] = []
    errors: dict[str, str] = {}
    if request.install_dependencies:
        installed, errors = install_custom_node_dependencies(settings, node_names=[name], all_enabled=False)
    return {"name": name, "path": str(target), "installed_dependencies": installed, "errors": errors}


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


@app.get("/api/runtime/capabilities")
async def runtime_capabilities() -> dict[str, Any]:
    snapshot = await _runtime_memory_snapshot()
    capabilities = _runtime_attention_capabilities()
    capabilities["comfy_running"] = snapshot.get("comfy_running", False)
    capabilities["comfy_system_stats"] = snapshot.get("comfy_system_stats", {})
    capabilities["active_runtime"] = settings.runtime.model_dump(mode="json")
    return capabilities


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


@app.get("/api/train-lora/templates")
async def train_lora_templates() -> dict[str, Any]:
    return build_train_lora_catalog(settings, await _runtime_memory_snapshot())


@app.get("/api/train-lora/devices")
async def train_lora_devices() -> dict[str, Any]:
    catalog = build_train_lora_catalog(settings, await _runtime_memory_snapshot())
    return {
        "device_profiles": catalog["device_profiles"],
        "recommended_device": catalog["recommended_device"],
        "detected_vram_gb": catalog["detected_vram_gb"],
    }


@app.get("/api/train-lora/pick-path")
async def train_lora_pick_path(kind: str = Query("dataset")) -> dict[str, str]:
    normalized = str(kind).lower()
    if normalized in {"base_model", "resume"}:
        title = "Select base model" if normalized == "base_model" else "Select LoRA checkpoint to continue training"
        patterns = [("Model files", "*.safetensors *.ckpt *.pt *.pth"), ("All files", "*.*")]
        try:
            path = await asyncio.to_thread(_pick_local_file_dialog, title, patterns)
        except Exception as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        return {"path": path}
    label = "output" if normalized == "output" else "dataset"
    try:
        path = await asyncio.to_thread(_pick_local_folder_dialog, f"Select Train LoRA {label} folder")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"path": path}


@app.get("/api/train-lora/pick-folder")
async def train_lora_pick_folder(kind: str = Query("dataset")) -> dict[str, str]:
    return await train_lora_pick_path(kind)


@app.post("/api/train-lora/start")
async def train_lora_start(plan: str = Form("{}"), files: list[UploadFile] = File(default=[])) -> dict[str, Any]:
    try:
        parsed_plan = json.loads(plan or "{}")
        if not isinstance(parsed_plan, dict):
            raise ValueError("Train LoRA plan must be a JSON object.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Train LoRA plan: {exc}") from exc

    job_id = uuid.uuid4().hex[:12]
    saved_files = await _save_train_lora_uploads(job_id, files or [])
    try:
        job = build_train_lora_job(settings, job_id, parsed_plan, saved_files, await _runtime_memory_snapshot())
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    train_lora_jobs[job_id] = job
    if _truthy(parsed_plan.get("launch")) and (job.get("runner") or {}).get("available"):
        _launch_train_lora_job(job_id)
    return public_train_lora_job(train_lora_jobs[job_id])


@app.get("/api/train-lora/{job_id}")
async def train_lora_status(job_id: str) -> dict[str, Any]:
    job = train_lora_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train LoRA job not found.")
    return public_train_lora_job(job)


@app.get("/api/train-lora/{job_id}/log")
async def train_lora_log(job_id: str) -> dict[str, Any]:
    job = train_lora_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train LoRA job not found.")
    log_path = Path(str(job.get("log_path") or (job.get("paths") or {}).get("terminal_log") or ""))
    if not log_path.exists():
        return {"job_id": job_id, "text": "", "path": str(log_path)}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {"job_id": job_id, "text": text[-40000:], "path": str(log_path)}


@app.post("/api/train-lora/{job_id}/cancel")
async def train_lora_cancel(job_id: str) -> dict[str, Any]:
    job = train_lora_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Train LoRA job not found.")
    process: subprocess.Popen[Any] | None = job.get("_process")
    if process and process.poll() is None:
        try:
            process.terminate()
        except Exception:
            pass
    log_handle = job.pop("_log_handle", None)
    try:
        if log_handle:
            log_handle.close()
    except Exception:
        pass
    _update_train_lora_job(
        job_id,
        {
            "status": "cancelled",
            "progress": 100,
            "message": "Train LoRA job cancelled.",
            "error": "cancelled",
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        },
    )
    return public_train_lora_job(job)


@app.post("/api/extras/start")
async def extras_start(plan: str = Form("{}"), files: list[UploadFile] = File(default=[])) -> dict[str, Any]:
    try:
        parsed_plan = json.loads(plan or "{}")
        if not isinstance(parsed_plan, dict):
            raise ValueError("Extras plan must be a JSON object.")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid Extras plan: {exc}") from exc
    source_files = await _save_extras_uploads(files or [])
    job_id = uuid.uuid4().hex[:12]
    extras_jobs[job_id] = {
        "job_id": job_id,
        "prompt_id": None,
        "status": "queued",
        "progress": 0,
        "message": "Queued Extras job.",
        "outputs": [],
        "error": None,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "preset": "Extras",
        "workflow_id": parsed_plan.get("template") or parsed_plan.get("mode") or "extras",
    }
    asyncio.create_task(_run_extras_job(job_id, source_files, parsed_plan))
    return _public_extras_job(extras_jobs[job_id])


@app.get("/api/extras/{job_id}")
async def extras_status(job_id: str) -> dict[str, Any]:
    job = extras_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Extras job not found.")
    return _public_extras_job(job)


async def _run_generation_core(request: GenerateRequest, job_id: str | None = None) -> GenerateResponse:
    if not settings.runtime.auto_start_comfy and not await comfy.is_running():
        raise HTTPException(status_code=503, detail="ComfyUI runtime is not running.")

    try:
        _cancel_comfy_idle_release()
        _prepare_runtime_for_generation(request)
        runtime_changed = _apply_runtime_options(request.runtime)
        _resolve_generation_seed(request)
        if (runtime_changed or comfy.runtime_changed_since_start()) and await comfy.is_running():
            if job_id:
                _update_generation_job(job_id, {"status": "starting", "progress": 2, "message": "Restarting ComfyUI runtime"}, force=True)
            cleanup_embedded_comfy_artifacts()
            await comfy.restart()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 4, "message": "Resolving generation assets"})
        assets = resolve_generation_assets(settings, request)
        _ensure_ltx_default_distilled_loras(request, assets)
        _ensure_wan_4step_loras(request, assets)
        _ensure_qwen_edit_lightning_lora(request, assets)
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
        base_video_name = _prepare_base_video(request)
        reference_image_name = reference_image_names[0] if reference_image_names else None
        reference_end_image_name = reference_image_names[1] if len(reference_image_names) > 1 else None
        mask_image_name = _prepare_mask_image(request)
        controlnet_image_name = _prepare_controlnet_image(request)
        if reference_image_name:
            assets["reference_image"] = reference_image_name
        if base_video_name:
            assets["base_video"] = base_video_name
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
        if request.preset.lower() == "wan" and not request.workflow_id:
            workflow_path = None
        if base_video_name and not request.workflow_id and request.preset.lower() in {"wan", "ltx"}:
            workflow_path = _find_video2video_workflow(request.preset)
            if not workflow_path:
                raise ValueError(f"{request.preset} video2video requires a compatible V2V workflow with LoadVideo/VHS_LoadVideo.")
        if request.preset.lower() == "qwen" and request.activity == "img2img" and reference_image_name:
            request.workflow_override = None
            workflow_path = None

        if job_id:
            _update_generation_job(job_id, {"status": "starting", "progress": 6, "message": "Starting embedded ComfyUI"}, force=True)
        await comfy.ensure_running()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 7, "message": "Reading Comfy object registry"})
        object_info = await comfy.object_info()

        if request.workflow_override:
            if job_id:
                _update_generation_job(job_id, {"status": "building", "progress": 9, "message": "Building visual workflow"})
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
                _update_generation_job(job_id, {"status": "building", "progress": 9, "message": "Patching workflow"})
            prompt = workflow_registry.load_api_workflow(workflow_path, request, object_info, assets=assets)
        else:
            checkpoint_name = assets.get("primary_model") or Path(request.model_path or request.model_name or "").name
            if not checkpoint_name:
                raise ValueError("No model selected.")
            if job_id:
                _update_generation_job(job_id, {"status": "building", "progress": 9, "message": "Building default workflow"})
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
                if checkpoint_name.lower().endswith(".gguf"):
                    raise ValueError("LTX img2vid default requires an LTX checkpoint file. GGUF workflows can still be loaded explicitly.")
                prompt = build_basic_ltx_img2video_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    reference_image_name,
                    reference_end_image_name=reference_end_image_name,
                    text_projection_name=assets.get("text_projection"),
                    audio_vae_name=assets.get("audio_vae"),
                    video_vae_name=assets.get("video_vae") or assets.get("vae"),
                    latent_upscale_name=assets.get("latent_upscale"),
                    video_combine_node=_available_comfy_node(object_info, "VHS_VideoCombine"),
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
                if not assets.get("clip_vision"):
                    raise ValueError("WAN 2.2 requires clip_vision_h.safetensors or a compatible CLIP Vision encoder in models/clip_vision.")
                wan_first_last_node = None
                if reference_end_image_name:
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
                    reference_end_image_name=reference_end_image_name,
                    first_last_frame_node=wan_first_last_node,
                    clip_vision_name=assets.get("clip_vision"),
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
                flux_family = assets.get("flux_family") or ""
                is_flux2 = str(flux_family).startswith("flux2")
                if not is_flux2 and not clip_l_name:
                    raise ValueError("Flux requires clip_l.safetensors in models/text_encoders.")
                if not text_encoder_name:
                    if flux_family == "flux2_dev":
                        raise ValueError("Flux.2 Dev requires mistral_3_small_flux2_bf16.safetensors in models/text_encoders.")
                    if flux_family == "flux2_klein_9b":
                        raise ValueError("Flux.2 Klein 9B requires qwen_3_8b_fp8mixed.safetensors in models/text_encoders.")
                    if flux_family == "flux2_klein_4b":
                        raise ValueError("Flux.2 Klein 4B requires qwen_3_4b.safetensors in models/text_encoders.")
                    raise ValueError("Flux requires a T5 text encoder in models/text_encoders.")
                if not vae_name:
                    if is_flux2:
                        raise ValueError("Flux.2 requires flux2-vae.safetensors or a compatible Flux.2 VAE in models/vae.")
                    raise ValueError("Flux requires an AE/Flux VAE in models/vae.")
                prompt = build_basic_flux_workflow(
                    request,
                    checkpoint_name,
                    clip_l_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
                    mask_image_name=mask_image_name,
                    flux_family=flux_family,
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
        _apply_output_prefixes(prompt, request)

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
            _update_generation_job(job_id, {"queue_position": 1, "message": "VRAM lock acquired."}, force=True)
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
        "_queued_monotonic": time.monotonic(),
    }
    _console_generation(generation_jobs[job_id], force=True)
    asyncio.create_task(_run_generation_job(job_id, request))
    return _public_generation_job(generation_jobs[job_id])


@app.get("/api/generate/{job_id}")
async def generate_status(job_id: str) -> dict[str, Any]:
    job = generation_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Generation job not found.")
    return _public_generation_job(job)


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
    return _public_generation_job(job)


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
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mkv", ".mov", ".avi"}:
            continue
        relative = path.relative_to(settings.output_dir).as_posix()
        url_path = quote(relative, safe="/")
        media_type = "video" if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".avi"} else "image"
        metadata = _read_output_metadata(path)
        items.append(
            {
                "title": path.name,
                "filename": path.name,
                "path": str(path),
                "relative_path": relative,
                "folder": path.parent.relative_to(settings.output_dir).as_posix() if path.parent != settings.output_dir else "",
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


@app.get("/app/{full_path:path}")
async def react_app(full_path: str = "") -> FileResponse:
    index = react_dist / "index.html"
    if not index.exists():
        raise HTTPException(status_code=404, detail="React frontend build not found. Run npm run build in frontend/.")
    return FileResponse(index, headers={"Cache-Control": "no-store, max-age=0", "Pragma": "no-cache"})


@app.get("/app")
async def react_app_root() -> RedirectResponse:
    return RedirectResponse(url="/app/", status_code=307)
