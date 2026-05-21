from __future__ import annotations

import asyncio
import base64
import json
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
from .comfy_client import ComfyClient
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
    RuntimeHealth,
    WorkflowSaveRequest,
)
from .templates import ensure_templates_file, load_templates
from .workflows import (
    WorkflowRegistry,
    build_basic_anima_workflow,
    build_basic_flux_workflow,
    build_basic_ltx_img2video_workflow,
    build_basic_qwen_image_workflow,
    build_basic_sd_workflow,
    build_basic_wan_i2video_workflow,
    convert_ui_to_api,
    detect_workflow_format,
    patch_workflow,
)


settings = load_settings()
generation_jobs: dict[str, dict[str, Any]] = {}
download_jobs: dict[str, dict[str, Any]] = {}

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


def _prepare_reference_image(request: GenerateRequest) -> str | None:
    value = (request.img2img.reference_image or "").strip()
    if request.activity != "img2img" or not value:
        return None

    if value.startswith("data:image/"):
        return _write_input_data_image(value, "nexus_reference")

    source: Path | None = None
    if value.startswith("/outputs/") or "/outputs/" in value:
        relative = unquote(value.split("/outputs/", 1)[1]).lstrip("/\\")
        source = (settings.output_dir / relative).resolve()
    else:
        candidate = Path(value)
        if candidate.exists():
            source = candidate.resolve()

    if not source or not source.exists():
        raise ValueError("Reference image could not be resolved.")

    suffix = source.suffix.lower() if source.suffix else ".png"
    filename = f"nexus_reference_{uuid.uuid4().hex[:10]}{suffix}"
    target = settings.input_dir / filename
    shutil.copy2(source, target)
    return filename


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
        value = (request.img2img.reference_image or "").strip()
    if not value:
        return None

    if value.startswith("data:image/"):
        return _write_input_data_image(value, "nexus_controlnet")

    source: Path | None = None
    if value.startswith("/outputs/") or "/outputs/" in value:
        relative = unquote(value.split("/outputs/", 1)[1]).lstrip("/\\")
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


def _generation_metadata(request: GenerateRequest) -> dict[str, Any]:
    controlnet = request.controlnet.model_dump()
    if controlnet.get("image"):
        image_value = str(controlnet.get("image") or "")
        controlnet["image"] = "embedded" if image_value.startswith("data:image/") else Path(image_value).name
    return {
        "prompt": request.prompt,
        "negative": request.negative_prompt,
        "model": request.model_name or Path(request.model_path or "").name,
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
        "loras": request.loras,
        "controlnet": controlnet,
    }


def _annotate_output_metadata(outputs: list[dict[str, Any]], request: GenerateRequest) -> None:
    metadata = _generation_metadata(request)
    for output in outputs:
        relative = str(output.get("path") or output.get("filename") or "")
        if not relative:
            continue
        path = (settings.output_dir / relative).resolve()
        if not path.exists() or not path.is_relative_to(settings.output_dir.resolve()):
            continue
        try:
            path.with_suffix(path.suffix + ".nexus.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
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
                pnginfo.add_text("nexus_bta", json.dumps(metadata, ensure_ascii=False))
                image.save(path, pnginfo=pnginfo)
        except Exception:
            continue


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
    for pattern in ("nexus_reference_*", "nexus_mask_*", "nexus_controlnet_*"):
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
    cleanup_embedded_comfy_artifacts()


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
        object_info = {}
        try:
            object_info = await comfy.object_info()
        except Exception:
            object_info = {}
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
        object_info = {}
        try:
            object_info = await comfy.object_info()
        except Exception:
            object_info = {}
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
    object_info = {}
    try:
        object_info = await comfy.object_info()
    except Exception:
        object_info = {}
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
    object_info = {}
    try:
        object_info = await comfy.object_info()
    except Exception:
        object_info = {}

    install_analysis = workflow_registry.analyze_workflow(
        workflow,
        object_info=object_info,
        install_dependencies=True,
    )

    if restart_comfy and install_analysis.dependencies_installed:
        cleanup_embedded_comfy_artifacts()
        await comfy.restart()
        try:
            object_info = await comfy.object_info()
        except Exception:
            object_info = {}
        refreshed = workflow_registry.analyze_workflow(workflow, object_info=object_info)
        refreshed.dependencies_installed = install_analysis.dependencies_installed
        refreshed.dependency_errors = install_analysis.dependency_errors
        return refreshed.model_dump(mode="json")

    return install_analysis.model_dump(mode="json")


@app.get("/api/comfy/object-info")
async def comfy_object_info() -> dict[str, Any]:
    try:
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
        if job_id:
            _update_generation_job(job_id, {"status": "starting", "progress": 1, "message": "Starting embedded ComfyUI"}, force=True)
        await comfy.ensure_running()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 3, "message": "Reading Comfy object registry"})
        object_info = await comfy.object_info()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 5, "message": "Resolving selected models and template assets"})
        assets = resolve_generation_assets(settings, request)
        reference_image_name = _prepare_reference_image(request)
        mask_image_name = _prepare_mask_image(request)
        controlnet_image_name = _prepare_controlnet_image(request)
        if reference_image_name:
            assets["reference_image"] = reference_image_name
        if mask_image_name:
            assets["mask_image"] = mask_image_name
        if controlnet_image_name:
            assets["controlnet_image"] = controlnet_image_name
        if assets.get("primary_model") and not request.model_name:
            request.model_name = assets["primary_model"]
        workflow_path = workflow_registry.find(request.workflow_id, request.preset)

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
                    raise ValueError("Anima requires a Qwen text encoder in models/text_encoders.")
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
                if not reference_image_name:
                    raise ValueError("LTX default generation requires a reference image for img2vid.")
                text_encoder_name = assets.get("text_encoder")
                if not text_encoder_name:
                    raise ValueError("LTX requires a Gemma text encoder in models/text_encoders.")
                if checkpoint_name.lower().endswith(".gguf"):
                    raise ValueError("LTX img2vid default requires an LTX checkpoint file. GGUF workflows can still be loaded explicitly.")
                prompt = build_basic_ltx_img2video_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    reference_image_name,
                    audio_vae_name=assets.get("audio_vae"),
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
                prompt = build_basic_wan_i2video_workflow(
                    request,
                    high_model_name,
                    low_model_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name,
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
                )

        def progress_callback(update: dict[str, Any]) -> None:
            if job_id:
                _update_generation_job(job_id, update)

        prompt_id, outputs = await comfy.run_workflow(prompt, progress_callback=progress_callback)
        _annotate_output_metadata(outputs, request)
        _cleanup_generation_temp()
        return GenerateResponse(
            job_id=prompt_id,
            prompt_id=prompt_id,
            status="completed",
            message="Generation completed.",
            outputs=outputs,
        )
    except Exception as exc:
        _cleanup_generation_temp()
        if job_id:
            _update_generation_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)}, force=True)
        raise


async def _run_generation_job(job_id: str, request: GenerateRequest) -> None:
    try:
        response = await _run_generation_core(request, job_id=job_id)
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
    except Exception as exc:
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


@app.post("/api/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest) -> GenerateResponse:
    try:
        return await _run_generation_core(request)
    except HTTPException:
        raise
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
                "modified": path.stat().st_mtime,
            }
        )
        if len(items) >= 200:
            break
    return items


@app.get("/")
async def root() -> dict[str, str]:
    return {"name": "Nexus BTA Backend", "status": "ok"}


@app.get("/ui")
async def ui() -> FileResponse:
    return FileResponse(settings.project_root / "index.html")


@app.get("/index.html")
async def index_html() -> FileResponse:
    return FileResponse(settings.project_root / "index.html")
