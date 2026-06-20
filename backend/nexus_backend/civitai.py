from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from time import monotonic
from time import sleep
from typing import Any, Callable

from .config import NexusSettings


API_HOSTS = ("https://civitai.red", "https://civitai.com")


def search_civitai_models(
    settings: NexusSettings | None = None,
    query: str = "",
    tag: str = "",
    token: str | None = None,
    types: str = "",
    base_model: str = "",
    sort: str = "Newest",
    period: str = "AllTime",
    nsfw: bool = True,
    limit: int = 24,
    cursor: str | None = None,
) -> dict[str, Any]:
    raw_query = str(query or "").strip()
    query = _normalize_search_query(query)
    tag = _normalize_tag(tag)
    if raw_query.startswith("#") and not tag:
        tag = _normalize_tag(raw_query)
        query = ""
    base_model = _civitai_base_model(base_model)
    params: dict[str, Any] = {
        "limit": max(1, min(int(limit or 24), 200)),
        "sort": sort or "Newest",
        "period": period or "AllTime",
        "nsfw": "true" if nsfw else "false",
    }
    if query:
        params["query"] = query
    if tag:
        params["tag"] = tag
    if types:
        params["types"] = types
    if base_model:
        params["baseModels"] = base_model
    if cursor:
        params["cursor"] = cursor
    try:
        data = _get_json_any_host(f"/api/v1/models?{urllib.parse.urlencode(params)}", token)
    except ValueError as exc:
        if cursor and _is_invalid_cursor_error(exc):
            return {"items": [], "metadata": {"cursorRejected": True}}
        raise
    items = data.get("items") or []
    if query and not items:
        for fallback_params in _fallback_search_params(query, params):
            data = _get_json_any_host(f"/api/v1/models?{urllib.parse.urlencode(fallback_params)}", token)
            items = data.get("items") or []
            if items:
                break
    if tag and not items:
        for fallback_params in _tag_fallback_search_params(tag, params):
            data = _get_json_any_host(f"/api/v1/models?{urllib.parse.urlencode(fallback_params)}", token)
            items = data.get("items") or []
            if items:
                break
    visible_items = [item for item in items if isinstance(item, dict)]
    if not nsfw:
        visible_items = [item for item in visible_items if _item_visible_when_mature_hidden(item)]
    metadata = _sanitize_search_metadata(data.get("metadata") or {}, params, len(visible_items))
    return {
        "items": [_normalize_model_item(item, settings=settings, token=token) for item in visible_items],
        "metadata": metadata,
    }


def _normalize_search_query(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lstrip("#"))


def _normalize_tag(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", str(value or "").strip().lstrip("#"))
    return cleaned.replace("_", "-")


def _fallback_search_query(value: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[_#,;:]+", " ", value)).strip()


def _fallback_search_params(query: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    fallbacks: list[dict[str, Any]] = []
    fallback_query = _fallback_search_query(query)
    if fallback_query and fallback_query != query:
        fallbacks.append({**params, "query": fallback_query})
    tag = fallback_query or query
    if tag:
        normalized_tag = re.sub(r"\s+", "-", tag.strip().lstrip("#"))
        for candidate in dict.fromkeys((tag, normalized_tag)):
            if candidate:
                fallback = {key: value for key, value in params.items() if key != "query"}
                fallback["tag"] = candidate
                fallbacks.append(fallback)
    return fallbacks


def _tag_fallback_search_params(tag: str, params: dict[str, Any]) -> list[dict[str, Any]]:
    query = _fallback_search_query(tag.replace("-", " "))
    if not query:
        return []
    fallback = {key: value for key, value in params.items() if key != "tag"}
    fallback["query"] = query
    return [fallback]


def _is_invalid_cursor_error(error: Exception) -> bool:
    text = str(error).lower()
    return "invalid cursor" in text or "expected 2 value" in text


def _cursor_is_numeric_offset(cursor: Any) -> bool:
    return bool(re.fullmatch(r"\d+", str(cursor or "").strip()))


def _sanitize_search_metadata(metadata: dict[str, Any], params: dict[str, Any], visible_count: int) -> dict[str, Any]:
    cleaned = dict(metadata or {})
    limit = max(1, min(int(params.get("limit") or 24), 200))
    next_cursor = cleaned.get("nextCursor")
    if visible_count < limit or _cursor_is_numeric_offset(next_cursor):
        cleaned.pop("nextCursor", None)
        cleaned.pop("nextPage", None)
    return cleaned


MATURE_TAGS = {
    "18+",
    "adult",
    "boobs",
    "explicit",
    "genital",
    "hentai",
    "mature",
    "naked",
    "nsfl",
    "nsfw",
    "nude",
    "nudity",
    "porn",
    "pussy",
    "sex",
    "sexual",
    "vagina",
}


def _mature_flag(value: Any) -> bool:
    if value is None or value == "":
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return int(value) > 2
    if isinstance(value, list):
        return any(_mature_flag(item) for item in value)
    text = str(value).strip().lower()
    if not text or text in {"0", "1", "false", "none", "safe", "sfw"}:
        return False
    return any(token in text for token in ("mature", "nsfw", "racy", "x", "xxx", "adult", "explicit"))


def _mature_text(value: Any) -> bool:
    tokens = {str(token or "").strip().lower() for token in (value if isinstance(value, list) else [value])}
    for token in tokens:
        if not token:
            continue
        normalized = re.sub(r"[^a-z0-9+]+", " ", token).strip()
        words = set(normalized.split())
        if token in MATURE_TAGS or normalized in MATURE_TAGS or words.intersection(MATURE_TAGS):
            return True
    return False


def _image_is_mature(image: dict[str, Any]) -> bool:
    return _mature_flag(image.get("nsfw")) or _mature_flag(image.get("needsReview")) or _mature_flag(image.get("nsfwLevel"))


def _item_visible_when_mature_hidden(item: dict[str, Any]) -> bool:
    if _mature_flag(item.get("nsfw")) or _mature_text(item.get("tags") or []) or _mature_text(item.get("name") or ""):
        return False
    images = [image for version in item.get("modelVersions") or [] for image in (version.get("images") or []) if isinstance(image, dict)]
    if not images:
        return True
    return any(not _image_is_mature(image) for image in images)


def _civitai_base_model(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = re.sub(r"[^a-z0-9]+", "", raw.lower())
    aliases = {
        "sd": "SD 1.5",
        "sd15": "SD 1.5",
        "sdxl": "SDXL 1.0",
        "flux": "Flux.1 D",
        "qwen": "Qwen",
        "qwenimage": "Qwen Image",
        "zimageturbo": "ZImageTurbo",
        "zimage": "ZImageBase",
        "zimagebase": "ZImageBase",
        "zimg": "ZImageTurbo",
        "wan": "WAN 2.2",
        "wan22": "WAN 2.2",
        "ltx": "LTXV 2.3",
        "ltx23": "LTXV 2.3",
    }
    return aliases.get(normalized, raw)


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
    resolved = {
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
        "preview": _preview_url(version, token=token),
        "previews": _preview_media(version, token=token),
        "description": model.get("description") or version.get("description") or "",
        "creator": (model.get("creator") or {}).get("username") if isinstance(model.get("creator"), dict) else "",
        "nsfw": bool(model.get("nsfw")),
        "nsfw_level": model.get("nsfwLevel"),
        "tags": model.get("tags") or [],
        "stats": model.get("stats") or {},
    }
    return _with_installed_state(resolved, settings)


def _normalize_model_item(item: dict[str, Any], settings: NexusSettings | None = None, token: str | None = None) -> dict[str, Any]:
    versions = item.get("modelVersions") or []
    normalized_versions: list[dict[str, Any]] = []
    for version in versions[:6]:
        if not isinstance(version, dict):
            continue
        primary = _primary_file(version)
        version_id = version.get("id")
        version_data = {
                "id": version_id,
                "name": version.get("name") or "",
                "base_model": version.get("baseModel") or "",
                "download_url": primary.get("downloadUrl") or (f"https://civitai.red/api/download/models/{version_id}" if version_id else ""),
                "file_name": primary.get("name") or "",
                "file_size_kb": primary.get("sizeKB") or 0,
                "trained_words": version.get("trainedWords") or [],
                "preview": _preview_url(version, token=token),
                "previews": _preview_media(version, token=token),
                "description": version.get("description") or "",
                "nsfw_level": version.get("nsfwLevel"),
                "url": f"https://civitai.red/models/{item.get('id')}?modelVersionId={version_id}" if version_id else f"https://civitai.red/models/{item.get('id')}",
        }
        if settings:
            kind = _target_kind(item.get("type"), str(version_data.get("file_name") or ""), "auto")
            target_preset = _preset_from_base_model(version_data.get("base_model"))
            version_data = _with_installed_state({**version_data, "model_type": item.get("type"), "target_folder": str(_target_dir(settings, kind, target_preset))}, settings)
        normalized_versions.append(version_data)
    preview = ""
    for candidate in normalized_versions:
        if candidate.get("preview"):
            preview = str(candidate.get("preview") or "")
            break
    normalized = {
        "id": item.get("id"),
        "name": item.get("name") or "Civitai model",
        "type": item.get("type") or "Unknown",
        "nsfw": bool(item.get("nsfw")),
        "nsfw_level": item.get("nsfwLevel"),
        "poi": bool(item.get("poi")),
        "creator": (item.get("creator") or {}).get("username") if isinstance(item.get("creator"), dict) else "",
        "description": item.get("description") or "",
        "tags": item.get("tags") or [],
        "stats": item.get("stats") or {},
        "versions": normalized_versions,
        "preview": preview,
        "url": f"https://civitai.red/models/{item.get('id')}",
    }
    if normalized_versions and normalized_versions[0].get("installed"):
        normalized["installed"] = True
        normalized["relative_path"] = normalized_versions[0].get("relative_path", "")
        normalized["path"] = normalized_versions[0].get("path", "")
    return normalized


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
    target = target_dir / filename
    if target.exists() and target.stat().st_size > 0:
        preview_path = _save_civitai_preview(target, resolved, token, save_preview, progress_callback)
        return _with_installed_state({
            **resolved,
            "target_kind": kind,
            "path": str(target),
            "relative_path": _safe_relative(target, settings.project_root),
            "preview_path": str(preview_path) if preview_path else "",
            "preview_relative_path": _safe_relative(preview_path, settings.project_root) if preview_path else "",
        }, settings)
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
    expected_bytes = int(float(resolved.get("file_size_kb") or 0) * 1024)
    _download_file(download_url, target, token, progress_callback=progress_callback, expected_bytes=expected_bytes)

    preview_path = _save_civitai_preview(target, resolved, token, save_preview, progress_callback)

    return {
        **resolved,
        "target_kind": kind,
        "path": str(target),
        "relative_path": _safe_relative(target, settings.project_root),
        "preview_path": str(preview_path) if preview_path else "",
        "preview_relative_path": _safe_relative(preview_path, settings.project_root) if preview_path else "",
        "installed": True,
        "already_downloaded": True,
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
    api_paths = [_api_path_with_token(path, token)]
    if token and api_paths[0] != path:
        # Some Civitai endpoints intermittently 500 with token in the query string.
        # Keep the Authorization header and retry the original path before failing.
        api_paths.append(path)
    for api_path in dict.fromkeys(api_paths):
        for host in API_HOSTS:
            for attempt in range(2):
                try:
                    request = urllib.request.Request(host + api_path, headers={**_headers(token), "Accept": "application/json,*/*"})
                    with urllib.request.urlopen(request, timeout=30) as response:
                        return json.loads(response.read().decode("utf-8"))
                except urllib.error.HTTPError as exc:
                    detail = _http_error_detail(exc)
                    errors.append(f"{urllib.parse.urlparse(host).netloc}: HTTP {exc.code}{f' - {detail}' if detail else ''}")
                    if exc.code < 500:
                        break
                    if attempt == 0:
                        sleep(0.35)
                        continue
                    break
                except Exception as exc:
                    errors.append(f"{urllib.parse.urlparse(host).netloc}: {type(exc).__name__}")
                    break
    raise ValueError(_friendly_civitai_error(errors))


def _http_error_detail(error: urllib.error.HTTPError) -> str:
    try:
        payload = error.read(600).decode("utf-8", errors="ignore").strip()
    except Exception:
        return ""
    if not payload:
        return ""
    try:
        data = json.loads(payload)
        payload = str(data.get("message") or data.get("detail") or payload)
    except Exception:
        pass
    return re.sub(r"\s+", " ", payload)[:180]


def _friendly_civitai_error(errors: list[str]) -> str:
    text = "; ".join(error for error in errors if error)
    if "HTTP 401" in text or "HTTP 403" in text:
        return "Civitai blocked this request. Check the API key/token permissions and mature-content settings."
    if "HTTP 429" in text:
        return "Civitai rate-limited the request. Wait a moment and retry."
    if "HTTP 500" in text or "HTTP 502" in text or "HTTP 503" in text or "HTTP 504" in text:
        return "Civitai is temporarily unstable for this search/filter. Retry, or adjust the filter if it keeps happening."
    return text[:300] or "Civitai request failed."


def _api_path_with_token(path: str, token: str | None) -> str:
    cleaned_token = str(token or "").strip()
    if not cleaned_token:
        return path
    parsed = urllib.parse.urlsplit(path)
    query = urllib.parse.parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key == "token" for key, _ in query):
        query.append(("token", cleaned_token))
    return urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, parsed.path, urllib.parse.urlencode(query), parsed.fragment))


def _download_file(
    url: str,
    target: Path,
    token: str | None,
    progress_callback: ProgressCallback | None = None,
    expected_bytes: int = 0,
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
                    total = _response_total_bytes(response, resume_from) or expected_bytes
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
                    if expected_bytes and downloaded <= 0:
                        raise ValueError("Civitai returned an empty response for a non-empty model file. Check token permissions and retry.")
                    if total and downloaded < total:
                        raise ValueError(f"Download interrupted at {downloaded}/{total} bytes.")
                    if expected_bytes and downloaded < max(1024, int(expected_bytes * 0.98)):
                        raise ValueError(f"Civitai download finished too small: {downloaded}/{expected_bytes} bytes.")
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


def _tokenized_preview_url(url: Any, token: str | None = None) -> str:
    text = str(url or "").strip()
    if not text:
        return ""
    if not token:
        return text
    host = urllib.parse.urlparse(text).netloc.lower()
    if "civitai." in host:
        return _with_token(text, token)
    return text


def _first_preview_value(image: dict[str, Any], keys: tuple[str, ...], token: str | None = None) -> str:
    for key in keys:
        value = image.get(key)
        if value:
            return _tokenized_preview_url(value, token)
    return ""


def _preview_url(version: dict[str, Any], token: str | None = None) -> str:
    previews = _preview_media(version, token=token)
    image = next((item for item in previews if str(item.get("type") or "").lower() != "video" and item.get("url")), None)
    source = image or next((item for item in previews if item.get("posterUrl") or item.get("thumbnailUrl") or item.get("url")), None)
    if not source:
        return ""
    return str(source.get("posterUrl") or source.get("thumbnailUrl") or source.get("url") or "")


def _preview_media(version: dict[str, Any], token: str | None = None) -> list[dict[str, Any]]:
    previews: list[dict[str, Any]] = []
    for image in version.get("images") or []:
        if not isinstance(image, dict):
            continue
        mime = str(image.get("mimeType") or image.get("type") or "").lower()
        video = image.get("video") if isinstance(image.get("video"), dict) else {}
        video_url = _tokenized_preview_url(
            image.get("lowResVideoUrl")
            or image.get("lowresVideoUrl")
            or video.get("lowResUrl")
            or video.get("lowresUrl")
            or image.get("videoUrl")
            or image.get("video_url")
            or video.get("url")
            or "",
            token,
        )
        poster_url = _first_preview_value(
            image,
            ("thumbnailUrl", "thumbUrl", "imageUrl", "lowResUrl", "previewUrl", "url"),
            token=token,
        )
        url = _tokenized_preview_url(image.get("url") or poster_url or video_url, token)
        if not url and not poster_url and not video_url:
            continue
        lower_url = url.lower()
        media_type = "video" if video_url or "video" in mime or lower_url.endswith((".mp4", ".webm", ".mov")) else "image"
        if media_type == "video" and not poster_url and not lower_url.endswith((".mp4", ".webm", ".mov")):
            poster_url = url
        previews.append(
            {
                "url": url,
                "type": media_type,
                "thumbnailUrl": poster_url,
                "posterUrl": poster_url,
                "videoUrl": video_url,
                "lowResUrl": _tokenized_preview_url(image.get("lowResUrl") or image.get("url") or poster_url, token),
                "lowResVideoUrl": _tokenized_preview_url(video.get("lowResUrl") or video_url, token),
                "nsfw": image.get("nsfw") or image.get("needsReview"),
                "nsfwLevel": image.get("nsfwLevel"),
                "nsfw_level": image.get("nsfwLevel"),
                "width": image.get("width"),
                "height": image.get("height"),
            }
        )
    return previews


def _preview_download_source(resolved: dict[str, Any]) -> dict[str, str] | None:
    preview_url = str(resolved.get("preview") or "")
    for item in resolved.get("previews") or []:
        if not isinstance(item, dict):
            continue
        item_url = str(item.get("url") or "")
        video_url = str(item.get("videoUrl") or item.get("video_url") or item.get("lowResVideoUrl") or "")
        poster_url = str(item.get("posterUrl") or item.get("thumbnailUrl") or item.get("imageUrl") or item.get("lowResUrl") or item.get("previewUrl") or item_url)
        item_type = str(item.get("type") or "").lower()
        if item_type == "video" and video_url and (not preview_url or item_url == preview_url):
            return {"url": video_url, "type": "video", "poster": poster_url}
        if poster_url and not poster_url.lower().split("?", 1)[0].endswith((".mp4", ".webm", ".mov")):
            return {"url": poster_url, "type": "image"}
    if preview_url:
        suffix = Path(str(urllib.parse.urlparse(preview_url).path)).suffix.lower()
        media_type = "video" if suffix in {".mp4", ".webm", ".mov"} else "image"
        return {"url": preview_url, "type": media_type}
    return None


def _preview_suffix(url: str, is_video: bool) -> str:
    if is_video:
        return ".jpg"
    suffix = Path(str(urllib.parse.urlparse(url).path)).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return ".jpg"
    if suffix == ".png":
        return ".png"
    return ".jpg"


def _save_civitai_preview(
    target: Path,
    resolved: dict[str, Any],
    token: str | None,
    save_preview: bool,
    progress_callback: ProgressCallback | None = None,
) -> Path | None:
    preview_source = _preview_download_source(resolved)
    if not save_preview or not preview_source:
        return None
    preview_url = preview_source["url"]
    preview_is_video = preview_source["type"] == "video"
    preview_path = target.with_suffix(target.suffix + f".preview{_preview_suffix(preview_url, preview_is_video)}")
    if preview_path.exists() and preview_path.stat().st_size > 0:
        return preview_path
    try:
        if progress_callback:
            progress_callback({"status": "saving_preview", "progress": 100, "message": "Saving preview"})
        if preview_is_video:
            if _save_video_first_frame(preview_url, preview_path):
                return preview_path
            poster = preview_source.get("poster")
            if poster:
                poster_path = target.with_suffix(target.suffix + f".preview{_preview_suffix(poster, False)}")
                _download_file(poster, poster_path, token=None)
                return poster_path if poster_path.exists() and poster_path.stat().st_size > 0 else None
            return None
        _download_file(preview_url, preview_path, token=None)
        return preview_path if preview_path.exists() and preview_path.stat().st_size > 0 else None
    except Exception:
        return None


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
            ("illustrious", ("illustrious", "illustriousxl", "illustrious xl")),
            ("noobai", ("noobai", "noob ai", "noobai xl", "noobai-xl")),
            ("animagine", ("animagine", "animaginexl", "animagine xl")),
            ("pony", ("pony", "pony diffusion", "pony diffusion v6 xl")),
            ("sdxl", ("sdxl", "stable diffusion xl")),
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
        return normalized
    return fallback


def _preset_folder(preset: str | None) -> str:
    value = re.sub(r"[^a-z0-9]+", "", str(preset or "download").lower())
    return {
        "sd": "sd15",
        "sd15": "sd15",
        "xl": "sdxl",
        "sdxl": "sdxl",
        "illustrious": "illustrious",
        "noobai": "noobai",
        "animagine": "animagine",
        "pony": "pony",
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


def _with_installed_state(data: dict[str, Any], settings: NexusSettings) -> dict[str, Any]:
    filename = _safe_filename(str(data.get("file_name") or ""))
    target_folder = Path(str(data.get("target_folder") or ""))
    if not filename:
        return data
    target = _find_installed_asset(settings, data, filename, target_folder)
    if target.exists() and target.is_file():
        return {
            **data,
            "installed": True,
            "downloaded": True,
            "already_downloaded": True,
            "exists": True,
            "path": str(target),
            "relative_path": _safe_relative(target, settings.project_root),
            "installed_path": _safe_relative(target, settings.project_root),
        }
    return data


def _candidate_install_roots(settings: NexusSettings, data: dict[str, Any], target_folder: Path) -> list[Path]:
    model_type = data.get("model_type") or data.get("target_kind") or ""
    kind = _target_kind(model_type, str(data.get("file_name") or ""), "auto")
    roots: list[Path] = []
    if target_folder:
        roots.append(target_folder)
    roots.extend(settings.model_sources.get(kind, []))
    roots.append(settings.models_dir / kind)
    if kind == "checkpoints":
        roots.extend([settings.models_dir / "checkpoints", settings.models_dir / "diffusion_models", settings.models_dir / "unet"])
        roots.extend(settings.model_sources.get("diffusion_models", []))
        roots.extend(settings.model_sources.get("unet", []))
    elif kind == "loras":
        roots.append(settings.models_dir / "loras")
    elif kind == "controlnet":
        roots.extend([settings.models_dir / "controlnet", settings.models_dir / "model_patches"])
        roots.extend(settings.model_sources.get("model_patches", []))
    deduped: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        try:
            resolved = str(Path(root).resolve())
        except Exception:
            resolved = str(root)
        if resolved and resolved not in seen:
            seen.add(resolved)
            deduped.append(Path(root))
    return deduped


def _find_installed_asset(settings: NexusSettings, data: dict[str, Any], filename: str, target_folder: Path) -> Path:
    for root in _candidate_install_roots(settings, data, target_folder):
        direct = root / filename
        if direct.exists() and direct.is_file():
            return direct
    lower_filename = filename.lower()
    for root in _candidate_install_roots(settings, data, target_folder):
        if not root.exists() or not root.is_dir():
            continue
        try:
            for candidate in root.rglob(filename):
                if candidate.is_file() and candidate.name.lower() == lower_filename:
                    return candidate
        except Exception:
            continue
    return target_folder / filename if target_folder else settings.models_dir / filename


def _save_video_first_frame(url: str, target: Path) -> bool:
    if not shutil.which("ffmpeg"):
        return False
    urls = _download_candidates(url, None)
    for candidate in urls:
        tmp = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(urllib.parse.urlparse(candidate).path).suffix or ".mp4") as handle:
                tmp = Path(handle.name)
            _download_file(candidate, tmp, token=None)
            target.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(tmp), "-frames:v", "1", "-q:v", "3", str(target)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            return target.exists() and target.stat().st_size > 0
        except Exception:
            if target.exists():
                target.unlink(missing_ok=True)
        finally:
            if tmp and tmp.exists():
                tmp.unlink(missing_ok=True)
    return False


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
