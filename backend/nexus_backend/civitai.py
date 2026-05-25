from __future__ import annotations

import json
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from time import monotonic
from typing import Any, Callable

from .config import NexusSettings


API_HOSTS = ("https://civitai.red", "https://civitai.com")


def search_civitai_models(
    query: str = "",
    token: str | None = None,
    types: str = "",
    base_model: str = "",
    sort: str = "Newest",
    period: str = "AllTime",
    nsfw: bool = True,
    limit: int = 24,
    cursor: str | None = None,
) -> dict[str, Any]:
    params: dict[str, Any] = {
        "limit": max(1, min(int(limit or 24), 100)),
        "sort": sort or "Newest",
        "period": period or "AllTime",
        "nsfw": "true" if nsfw else "false",
    }
    if query:
        params["query"] = query
    if types:
        params["types"] = types
    if base_model:
        params["baseModels"] = base_model
    if cursor:
        params["cursor"] = cursor
    data = _get_json_any_host(f"/api/v1/models?{urllib.parse.urlencode(params)}", token)
    items = data.get("items") or []
    return {
        "items": [_normalize_model_item(item) for item in items if isinstance(item, dict)],
        "metadata": data.get("metadata") or {},
    }


def resolve_civitai_asset(settings: NexusSettings, url: str, token: str | None = None, target_kind: str = "auto", preset: str | None = None) -> dict[str, Any]:
    version_id, model_id = _ids_from_url(url)
    if version_id:
        version = _get_json_any_host(f"/api/v1/model-versions/{version_id}", token)
        model = _get_json_any_host(f"/api/v1/models/{version.get('modelId')}", token) if version.get("modelId") else {}
    elif model_id:
        model = _get_json_any_host(f"/api/v1/models/{model_id}", token)
        versions = model.get("modelVersions") or []
        if not versions:
            raise ValueError("No model versions found for this Civitai model.")
        version = versions[0]
        version_id = int(version["id"])
    else:
        raise ValueError("Paste a Civitai model or model version URL.")

    file_info = _primary_file(version)
    base_model = version.get("baseModel") or ""
    target_kind = _target_kind(model.get("type"), file_info.get("name", ""), target_kind)
    target_preset = _preset_from_base_model(base_model, preset)
    return {
        "model_id": model.get("id") or version.get("modelId"),
        "model_name": model.get("name") or "Civitai model",
        "model_type": model.get("type") or "Unknown",
        "version_id": version_id or version.get("id"),
        "version_name": version.get("name") or "",
        "base_model": base_model,
        "target_preset": target_preset or "",
        "file_name": file_info.get("name") or f"civitai_{version_id}.safetensors",
        "file_size_kb": file_info.get("sizeKB") or 0,
        "download_url": file_info.get("downloadUrl") or f"https://civitai.red/api/download/models/{version_id}",
        "url": f"https://civitai.red/models/{model.get('id') or version.get('modelId')}?modelVersionId={version_id or version.get('id')}",
        "target_kind": target_kind,
        "target_folder": str(_target_dir(settings, target_kind, target_preset)),
        "trained_words": version.get("trainedWords") or [],
        "preview": _preview_url(version),
        "previews": _preview_media(version),
        "description": model.get("description") or version.get("description") or "",
        "creator": (model.get("creator") or {}).get("username") if isinstance(model.get("creator"), dict) else "",
        "stats": model.get("stats") or {},
    }


def _normalize_model_item(item: dict[str, Any]) -> dict[str, Any]:
    versions = item.get("modelVersions") or []
    normalized_versions: list[dict[str, Any]] = []
    for version in versions[:6]:
        if not isinstance(version, dict):
            continue
        primary = _primary_file(version)
        version_id = version.get("id")
        normalized_versions.append(
            {
                "id": version_id,
                "name": version.get("name") or "",
                "base_model": version.get("baseModel") or "",
                "download_url": primary.get("downloadUrl") or (f"https://civitai.red/api/download/models/{version_id}" if version_id else ""),
                "file_name": primary.get("name") or "",
                "file_size_kb": primary.get("sizeKB") or 0,
                "trained_words": version.get("trainedWords") or [],
                "preview": _preview_url(version),
                "previews": _preview_media(version),
                "description": version.get("description") or "",
                "url": f"https://civitai.red/models/{item.get('id')}?modelVersionId={version_id}" if version_id else f"https://civitai.red/models/{item.get('id')}",
            }
        )
    preview = ""
    if normalized_versions:
        preview = normalized_versions[0].get("preview", "")
    return {
        "id": item.get("id"),
        "name": item.get("name") or "Civitai model",
        "type": item.get("type") or "Unknown",
        "nsfw": bool(item.get("nsfw")),
        "poi": bool(item.get("poi")),
        "creator": (item.get("creator") or {}).get("username") if isinstance(item.get("creator"), dict) else "",
        "description": item.get("description") or "",
        "tags": item.get("tags") or [],
        "stats": item.get("stats") or {},
        "versions": normalized_versions,
        "preview": preview,
        "url": f"https://civitai.red/models/{item.get('id')}",
    }


ProgressCallback = Callable[[dict[str, Any]], None]


def download_civitai_asset(
    settings: NexusSettings,
    url: str,
    token: str | None = None,
    target_kind: str = "auto",
    preset: str | None = None,
    save_preview: bool = True,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    resolved = resolve_civitai_asset(settings, url, token, target_kind=target_kind, preset=preset)
    kind = _target_kind(resolved.get("model_type"), resolved.get("file_name", ""), target_kind)
    target_dir = Path(str(resolved.get("target_folder") or _target_dir(settings, kind, preset)))
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = _safe_filename(str(resolved["file_name"]))
    target = _unique_path(target_dir / filename)
    download_url = str(resolved["download_url"])
    if progress_callback:
        progress_callback(
            {
                "status": "downloading",
                "model_name": resolved.get("model_name") or "Civitai model",
                "file_name": filename,
                "bytes_total": int(float(resolved.get("file_size_kb") or 0) * 1024),
                "progress": 0,
                "speed_bps": 0,
            }
        )
    _download_file(download_url, target, token, progress_callback=progress_callback)

    preview_path = None
    if save_preview and resolved.get("preview"):
        suffix = Path(str(urllib.parse.urlparse(str(resolved["preview"])).path)).suffix or ".png"
        preview_path = _unique_path(target.with_suffix(target.suffix + f".preview{suffix}"))
        try:
            if progress_callback:
                progress_callback({"status": "saving_preview", "progress": 100, "message": "Saving preview"})
            _download_file(str(resolved["preview"]), preview_path, token=None)
        except Exception:
            preview_path = None

    return {
        **resolved,
        "target_kind": kind,
        "path": str(target),
        "relative_path": _safe_relative(target, settings.project_root),
        "preview_path": str(preview_path) if preview_path else "",
        "preview_relative_path": _safe_relative(preview_path, settings.project_root) if preview_path else "",
    }


def _ids_from_url(url: str) -> tuple[int | None, int | None]:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qs(parsed.query)
    version = _int_or_none((query.get("modelVersionId") or query.get("versionId") or [None])[0])
    model = None
    match = re.search(r"/api/download/models/(\d+)", parsed.path)
    if match:
        version = int(match.group(1))
    match = re.search(r"/api/v1/model-versions/(\d+)", parsed.path)
    if match:
        version = int(match.group(1))
    match = re.search(r"/models/(\d+)", parsed.path)
    if match:
        model = int(match.group(1))
    return version, model


def _get_json_any_host(path: str, token: str | None) -> dict[str, Any]:
    errors: list[str] = []
    for host in API_HOSTS:
        try:
            request = urllib.request.Request(host + path, headers=_headers(token))
            with urllib.request.urlopen(request, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            errors.append(str(exc))
    raise ValueError("; ".join(errors) or "Civitai request failed.")


def _download_file(
    url: str,
    target: Path,
    token: str | None,
    progress_callback: ProgressCallback | None = None,
) -> None:
    urls = _download_candidates(url, token)
    last_error: Exception | None = None
    part = target.with_suffix(target.suffix + ".part")
    for candidate in urls:
        for attempt in range(4):
            try:
                resume_from = part.stat().st_size if part.exists() else 0
                headers = _headers(token)
                if resume_from:
                    headers["Range"] = f"bytes={resume_from}-"
                request = urllib.request.Request(candidate, headers=headers)
                with urllib.request.urlopen(request, timeout=180) as response:
                    status_code = int(getattr(response, "status", 200) or 200)
                    if resume_from and status_code == 200:
                        resume_from = 0
                        part.unlink(missing_ok=True)
                    total = _response_total_bytes(response, resume_from)
                    downloaded = resume_from
                    started = monotonic()
                    last_emit = started
                    mode = "ab" if resume_from else "wb"
                    with part.open(mode) as output:
                        first_chunk = True
                        while True:
                            chunk = response.read(4 * 1024 * 1024)
                            if not chunk:
                                break
                            if first_chunk:
                                _raise_if_error_payload(response, chunk)
                                first_chunk = False
                            output.write(chunk)
                            downloaded += len(chunk)
                            now = monotonic()
                            if progress_callback and (now - last_emit >= 0.5 or (total and downloaded >= total)):
                                elapsed = max(0.001, now - started)
                                progress_callback(
                                    {
                                        "status": "downloading",
                                        "bytes_downloaded": downloaded,
                                        "bytes_total": total,
                                        "progress": round((downloaded / total) * 100, 2) if total else None,
                                        "speed_bps": round((downloaded - resume_from) / elapsed),
                                        "message": f"Downloading via {urllib.parse.urlparse(candidate).netloc}",
                                    }
                                )
                                last_emit = now
                    if total and downloaded < total:
                        raise ValueError(f"Download interrupted at {downloaded}/{total} bytes.")
                    target.unlink(missing_ok=True)
                    part.replace(target)
                    if progress_callback:
                        progress_callback(
                            {
                                "status": "downloaded",
                                "bytes_downloaded": downloaded,
                                "bytes_total": total or downloaded,
                                "progress": 100,
                                "speed_bps": round(max(0, downloaded - resume_from) / max(0.001, monotonic() - started)),
                            }
                        )
                return
            except Exception as exc:
                last_error = exc
                if attempt < 3:
                    continue
        part.unlink(missing_ok=True)
        if target.exists():
            target.unlink(missing_ok=True)
    raise ValueError(str(last_error) if last_error else "Download failed.")


def _download_candidates(url: str, token: str | None) -> list[str]:
    parsed = urllib.parse.urlparse(url)
    variants = []
    if parsed.netloc:
        if "civitai." in parsed.netloc:
            variants.append(urllib.parse.urlunparse(parsed._replace(netloc="civitai.com")))
            variants.append(urllib.parse.urlunparse(parsed._replace(netloc="civitai.red")))
        variants.append(url)
    version_match = re.search(r"/api/download/models/(\d+)", parsed.path)
    if version_match:
        version_id = version_match.group(1)
        variants.append(f"https://civitai.com/api/download/models/{version_id}")
        variants.append(f"https://civitai.red/api/download/models/{version_id}")
    deduped: list[str] = []
    for item in variants:
        candidate = _with_token(item, token) if token and "civitai." in urllib.parse.urlparse(item).netloc else item
        if candidate not in deduped:
            deduped.append(candidate)
    return deduped


def _response_total_bytes(response: Any, resume_from: int = 0) -> int:
    content_range = response.headers.get("Content-Range") or ""
    match = re.search(r"/(\d+)\s*$", content_range)
    if match:
        return int(match.group(1))
    content_length = int(response.headers.get("Content-Length") or 0)
    return content_length + resume_from if resume_from and content_length else content_length


def _raise_if_error_payload(response: Any, chunk: bytes) -> None:
    content_type = str(response.headers.get("Content-Type") or "").lower()
    if "text/html" not in content_type and "application/json" not in content_type:
        return
    snippet = chunk[:800].decode("utf-8", errors="ignore").strip()
    if "error" in snippet.lower() or "login" in snippet.lower() or "unauthorized" in snippet.lower() or "<html" in snippet.lower():
        raise ValueError(f"Civitai returned {content_type or 'an error page'} instead of a model file: {snippet[:240]}")


def _headers(token: str | None) -> dict[str, str]:
    headers = {
        "User-Agent": "Mozilla/5.0 NexusBTA/0.1",
        "Accept": "application/octet-stream,*/*",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def _with_token(url: str, token: str) -> str:
    parsed = urllib.parse.urlparse(url)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key == "token" for key, _ in query):
        query.append(("token", token))
    return urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))


def _primary_file(version: dict[str, Any]) -> dict[str, Any]:
    files = version.get("files") or []
    if not files:
        return {}
    return next((file for file in files if file.get("primary")), files[0])


def _preview_url(version: dict[str, Any]) -> str:
    images = version.get("images") or []
    return str(images[0].get("url") or "") if images else ""


def _preview_media(version: dict[str, Any]) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for image in version.get("images") or []:
        if not isinstance(image, dict):
            continue
        url = str(image.get("url") or "")
        if not url:
            continue
        mime = str(image.get("mimeType") or image.get("type") or "").lower()
        lower_url = url.lower()
        media_type = "video" if "video" in mime or lower_url.endswith((".mp4", ".webm", ".mov")) else "image"
        previews.append(
            {
                "url": url,
                "type": media_type,
                "nsfw": image.get("nsfw") or image.get("needsReview"),
                "width": image.get("width"),
                "height": image.get("height"),
            }
        )
    return previews


def _target_kind(model_type: Any, filename: str, requested: str) -> str:
    if requested and requested != "auto":
        return requested
    lower_type = str(model_type or "").lower()
    lower_name = filename.lower()
    if lower_name.endswith(".json"):
        return "workflows"
    if "lora" in lower_type:
        return "loras"
    if "textualinversion" in lower_type or "embedding" in lower_type:
        return "embeddings"
    if "qwen_3_4b" in lower_name or "qwen3_4b" in lower_name:
        return "text_encoders"
    if lower_name == "ae.safetensors":
        return "vae"
    if "vae" in lower_type or "vae" in lower_name:
        return "vae"
    if any(token in lower_name for token in ("z_image", "z-image", "zimage")):
        return "diffusion_models"
    if "controlnet" in lower_type or "control" in lower_name:
        return "controlnet"
    if any(token in lower_name for token in ("biref", "rmbg", "rembg", "background", "inspy", "isnet", "ben2")):
        return "background_removal"
    if "upscaler" in lower_type or "upscale" in lower_name:
        return "upscale_models"
    if "motion" in lower_type:
        return "animatediff_models"
    return "checkpoints"


def _target_dir(settings: NexusSettings, kind: str, preset: str | None) -> Path:
    preset_folder = _preset_folder(preset)
    if kind == "workflows":
        return settings.workflows_dir
    if kind == "checkpoints":
        return settings.models_dir / "checkpoints" / preset_folder
    if kind == "loras":
        return settings.models_dir / "loras" / preset_folder if preset_folder != "download" else settings.models_dir / "loras"
    return settings.models_dir / kind


def _preset_from_base_model(base_model: Any, fallback: str | None = None) -> str | None:
    raw = str(base_model or "")
    normalized = re.sub(r"[^a-z0-9]+", "", raw.lower())
    if normalized:
        rules = (
            ("sd15", ("sd15", "sd1", "stable diffusion 1", "stable diffusion 1.5")),
            ("sdxl", ("sdxl", "stable diffusion xl", "pony")),
            ("flux", ("flux", "flux1")),
            ("qwen", ("qwen", "qwen image", "qwenimage")),
            ("zimage", ("zimage", "z image", "z-image", "zimage turbo", "z image turbo")),
            ("wan", ("wan", "wan2", "wan22", "wan21")),
            ("ltx", ("ltx", "ltx video", "ltxvideo", "ltx23")),
            ("anima", ("anima",)),
            ("lumina", ("lumina",)),
        )
        for preset, aliases in rules:
            if any(re.sub(r"[^a-z0-9]+", "", alias.lower()) in normalized for alias in aliases):
                return preset
    return fallback


def _preset_folder(preset: str | None) -> str:
    value = re.sub(r"[^a-z0-9]+", "", str(preset or "download").lower())
    return {
        "sd": "sd15",
        "sd15": "sd15",
        "xl": "sdxl",
        "sdxl": "sdxl",
        "ltx": "ltx",
        "anima": "anima",
        "wan": "wan",
        "flux": "flux",
        "qwen": "qwen",
        "zimage": "zimage",
        "zimageturbo": "zimage",
        "lumina": "lumina",
    }.get(value, value or "download")


def _safe_filename(value: str) -> str:
    name = Path(value).name
    return re.sub(r"[^a-zA-Z0-9._ -]+", "_", name).strip(" ._") or "civitai_asset.safetensors"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    counter = 1
    while True:
        candidate = path.with_name(f"{stem}_{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def _safe_relative(path: Path | None, root: Path) -> str:
    if path is None:
        return ""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
