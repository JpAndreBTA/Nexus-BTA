from __future__ import annotations

from pathlib import Path

from .config import NexusSettings
from .scanner import scan_models
from .schemas import GenerateRequest, ModelFile


def resolve_generation_assets(settings: NexusSettings, request: GenerateRequest) -> dict[str, str]:
    catalog = scan_models(settings, include_references=False)
    by_category = catalog.categories
    assets: dict[str, str] = {}

    selected_name = Path(request.model_path or request.model_name or "").name
    selected_model = _find_name(by_category, selected_name)
    if selected_model:
        assets["primary_model"] = _comfy_name(selected_model)
    elif selected_name:
        assets["primary_model"] = selected_name

    preset = request.preset.lower()
    if preset == "ltx":
        assets.update(_resolve_ltx(by_category, selected_name, request))
    elif preset == "wan":
        assets.update(_resolve_wan(by_category, selected_name, request))
    elif preset == "flux":
        assets.update(_resolve_flux(by_category, selected_name, request))
    elif preset == "anima":
        assets.update(_resolve_anima(by_category, selected_name, request))
    elif preset == "qwen":
        assets.pop("primary_model", None)
        assets.update(_resolve_qwen(by_category, selected_name, request))
    else:
        if "primary_model" not in assets:
            match = _first(by_category, ["checkpoints", "diffusion_models", "unet"], [])
            if match:
                assets["primary_model"] = match.name
        selected_vae = _selected_model_choice(by_category, request.vae)
        selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
        if selected_vae:
            assets["vae"] = _comfy_name(selected_vae)
        if selected_text_encoder:
            assets["text_encoder"] = _comfy_name(selected_text_encoder)

    assets.update(_resolve_controlnet(by_category, request))
    return {key: value for key, value in assets.items() if value}


def _resolve_controlnet(by_category: dict[str, list[ModelFile]], request: GenerateRequest) -> dict[str, str]:
    control = request.controlnet
    if not control.enabled:
        return {}
    preset = request.preset.lower()
    if preset not in {"sd", "sd15", "xl", "sdxl"}:
        return {}
    selected = _selected_model_choice(by_category, control.model)
    if selected and selected.category != "controlnet":
        selected = None
    control_type = str(control.type or "").lower()
    candidates = by_category.get("controlnet", [])
    model = selected
    if not model:
        model = _first_controlnet(candidates, preset, control_type)
    if not model:
        return {}
    return {"controlnet_model": _comfy_name(model)}


def _first_controlnet(items: list[ModelFile], preset: str, control_type: str) -> ModelFile | None:
    preset_tokens = ["sdxl", "xl"] if preset in {"xl", "sdxl"} else ["sd15", "sd1", "v11", "1.5"]
    type_tokens = {
        "openpose": ["openpose", "pose"],
        "pose": ["openpose", "pose"],
        "depth": ["depth"],
        "canny": ["canny"],
        "lineart": ["lineart", "line"],
        "tile": ["tile"],
    }.get(control_type, [control_type] if control_type else [])
    scored: list[tuple[int, str, ModelFile]] = []
    for item in items:
        haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
        if not any(token in haystack for token in preset_tokens):
            continue
        if type_tokens and not any(token in haystack for token in type_tokens):
            continue
        scored.append((len(item.name), item.name.lower(), item))
    if not scored:
        return None
    scored.sort(key=lambda row: row[:2])
    return scored[0][2]


def _resolve_flux(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    primary = _find_name(by_category, selected_name)
    if not primary:
        primary = (
            _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["flux", "q5"])
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["flux", "dev"])
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["flux"])
        )
    if primary:
        assets["primary_model"] = _comfy_name(primary)

    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    selected_vae = _selected_model_choice(by_category, request.vae)
    clip_l = (
        selected_text_encoder
        if selected_text_encoder and "clip_l" in " ".join([selected_text_encoder.name, selected_text_encoder.folder, selected_text_encoder.relative_path]).lower()
        else _first(by_category, ["text_encoders", "clip"], ["clip_l"])
    )
    t5 = (
        selected_text_encoder
        if selected_text_encoder and "clip_l" not in " ".join([selected_text_encoder.name, selected_text_encoder.folder, selected_text_encoder.relative_path]).lower()
        else None
    ) or (
        _first(by_category, ["text_encoders", "clip"], ["t5xxl", "fp8"])
        or _first(by_category, ["text_encoders", "clip"], ["t5", "fp8"])
        or _first(by_category, ["text_encoders", "clip"], ["t5"])
    )
    vae = (
        selected_vae
        or _first(by_category, ["vae"], ["flux_ae"])
        or _first(by_category, ["vae"], ["ae.safetensors"])
        or _first(by_category, ["vae"], ["flux", "vae"])
        or _first(by_category, ["vae"], ["ae"])
        or _first(by_category, ["vae"], ["flux"])
    )
    if clip_l:
        assets["flux_clip_l"] = _comfy_name(clip_l)
    if t5:
        assets["text_encoder"] = _comfy_name(t5)
    if vae:
        assets["vae"] = _comfy_name(vae)
    return assets


def _resolve_ltx(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    video_options = request.video or {}
    primary = _find_name(by_category, selected_name)
    if not primary:
        primary = (
            _first(by_category, ["checkpoints"], ["ltx"], exclude_extensions={".gguf"})
            or _first(by_category, ["checkpoints"], ["eros"], exclude_extensions={".gguf"})
            or _first(by_category, ["checkpoints"], ["sulphur"], exclude_extensions={".gguf"})
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["ltx-2.3"])
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["ltx"])
            or _first(by_category, ["checkpoints", "diffusion_models", "unet"], ["sulphur"])
        )
    if primary:
        assets["primary_model"] = _comfy_name(primary)

    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    if selected_text_encoder and _is_text_projection(selected_text_encoder):
        selected_text_encoder = None
    gemma = selected_text_encoder or _first(by_category, ["text_encoders", "clip"], ["gemma"])
    projection = _first(by_category, ["checkpoints", "text_encoders", "vae"], ["projection"])
    if gemma:
        assets["text_encoder"] = _comfy_name(gemma)
    if projection:
        assets["text_projection"] = _comfy_name(projection)

    selected_video_vae = _selected_asset(video_options.get("video_vae")) or _selected_asset(request.vae)
    selected_audio_vae = _selected_asset(video_options.get("audio_vae"))
    selected_upscale = _selected_asset(video_options.get("latent_upscale"))
    video_vae = _find_name(by_category, selected_video_vae)
    if video_vae and _is_ltx_preview_vae(video_vae):
        video_vae = None
    video_vae = video_vae or _first(by_category, ["vae"], ["video", "ltx"])
    audio_vae = _find_name(by_category, selected_audio_vae) or _first(by_category, ["vae"], ["audio", "ltx"])
    preview_vae = _first(by_category, ["vae"], ["taeltx"])
    if video_vae:
        assets["video_vae"] = _comfy_name(video_vae)
        assets["vae"] = _comfy_name(video_vae)
    if audio_vae:
        assets["audio_vae"] = _comfy_name(audio_vae)
    if preview_vae:
        assets["preview_vae"] = _comfy_name(preview_vae)

    upscale = _find_name(by_category, selected_upscale) or _first(by_category, ["latent_upscale_models", "upscale_models"], ["ltx", "spatial"])
    if upscale:
        assets["latent_upscale"] = _comfy_name(upscale)

    distilled_safe = _first(by_category, ["loras"], ["distilled", "condsafe"])
    distilled_large = _first(by_category, ["loras"], ["distilled", "384"])
    distilled_any = _first(by_category, ["loras"], ["distilled"])
    if distilled_safe:
        assets["distilled_lora_1"] = _comfy_name(distilled_safe)
    elif distilled_any:
        assets["distilled_lora_1"] = _comfy_name(distilled_any)
    if distilled_large:
        assets["distilled_lora_2"] = _comfy_name(distilled_large)

    ic_lora = _first(by_category, ["loras"], ["ic-lora", "cameraman"])
    if ic_lora:
        assets["ic_lora"] = _comfy_name(ic_lora)
    return assets


def _resolve_wan(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    selected = _find_name(by_category, selected_name)
    high_model = selected if selected and _is_wan_noise_model(selected, "high") else None
    low_model = selected if selected and _is_wan_noise_model(selected, "low") else None

    high_model = high_model or (
        _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan2.2", "high"])
        or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan", "high"])
    )
    low_model = low_model or (
        _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan2.2", "low"])
        or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan", "low"])
    )

    primary = high_model or selected or low_model
    if primary:
        assets["primary_model"] = _comfy_name(primary)
    if high_model:
        assets["wan_high_model"] = _comfy_name(high_model)
    if low_model:
        assets["wan_low_model"] = _comfy_name(low_model)

    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    selected_vae = _selected_model_choice(by_category, request.vae)
    text_encoder = (
        selected_text_encoder
        or
        _first(by_category, ["text_encoders", "clip"], ["umt5"])
        or _first(by_category, ["text_encoders", "clip"], ["t5"])
    )
    vae = selected_vae or _first(by_category, ["vae"], ["wan"])
    if text_encoder:
        assets["text_encoder"] = _comfy_name(text_encoder)
    if vae:
        assets["vae"] = _comfy_name(vae)
    return assets


def _resolve_anima(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    primary = _find_name(by_category, selected_name)
    if not primary:
        primary = _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["anima", "anime", "pencil"])
        if not primary:
            primary = _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["anima"])
    if primary:
        assets["primary_model"] = _comfy_name(primary)
    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    selected_vae = _selected_model_choice(by_category, request.vae)
    text_encoder = selected_text_encoder or _first(by_category, ["text_encoders", "clip"], ["qwen_3"]) or _first(
        by_category, ["text_encoders", "clip"], ["qwen"]
    )
    if text_encoder:
        assets["text_encoder"] = _comfy_name(text_encoder)
    vae = (
        selected_vae
        or _first(by_category, ["vae"], ["qwen", "image"])
        or _first(by_category, ["vae"], ["qwen"])
        or _first(by_category, ["vae"], ["anime", "kl-f8", "vae"])
    )
    if vae:
        assets["vae"] = _comfy_name(vae)
    return assets


def _resolve_qwen(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    has_reference = request.activity == "img2img" and bool((request.img2img.reference_image or "").strip())
    primary = _find_name(by_category, selected_name)
    if primary and _is_qwen_edit_model(primary) and not has_reference:
        primary = None
    if not primary:
        if has_reference:
            primary = (
                _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["qwen", "edit", "q4"])
                or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["qwen", "edit"])
            )
        primary = primary or (
            _first_qwen_base(by_category, ["qwen", "2512"])
            or _first_qwen_base(by_category, ["qwen", "image", "q4"])
            or _first_qwen_base(by_category, ["qwen", "image", "gguf"])
            or _first_qwen_base(by_category, ["qwen", "image"])
            or _first_qwen_base(by_category, ["qwen"])
        )
    if primary:
        assets["primary_model"] = _comfy_name(primary)
    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    text_encoder = (
        selected_text_encoder
        or _first(by_category, ["text_encoders", "clip"], ["qwen_2.5"])
        or _first(by_category, ["text_encoders", "clip"], ["qwen", "vl"])
        or _first(by_category, ["text_encoders", "clip"], ["qwen"])
    )
    if text_encoder:
        assets["text_encoder"] = _comfy_name(text_encoder)
    selected_vae = _selected_model_choice(by_category, request.vae)
    vae = (
        selected_vae
        or _first(by_category, ["vae"], ["qwen", "image"])
        or _first(by_category, ["vae"], ["qwen"])
    )
    if vae:
        assets["vae"] = _comfy_name(vae)
    return assets


def _selected_model_choice(by_category: dict[str, list[ModelFile]], value: str | None) -> ModelFile | None:
    name = Path(str(value or "")).name
    if not name or name.lower() in {"automatic", "auto", "none"}:
        return None
    return _find_name(by_category, name)


def _first_qwen_base(by_category: dict[str, list[ModelFile]], tokens: list[str]) -> ModelFile | None:
    lowered = [token.lower() for token in tokens]
    for category in ["checkpoints", "diffusion_models", "unet"]:
        for item in by_category.get(category, []):
            haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
            if all(token in haystack for token in lowered) and not _is_qwen_edit_model(item):
                return item
    return None


def _is_qwen_edit_model(item: ModelFile) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "qwen" in haystack and "edit" in haystack


def _is_ltx_preview_vae(item: ModelFile) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "taeltx" in haystack or "preview" in haystack


def _is_text_projection(item: ModelFile) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "projection" in haystack or "text_projection" in haystack or "proj" in haystack


def _comfy_name(item: ModelFile) -> str:
    relative = item.relative_path.replace("/", "\\")
    prefix = f"{item.category}\\"
    if relative.lower().startswith(prefix.lower()):
        name = relative[len(prefix) :]
        if item.category == "loras" and name.lower().startswith("ltx2\\"):
            return "ltx\\" + name.split("\\", 1)[1]
        return name
    return item.name


def _find_name(by_category: dict[str, list[ModelFile]], name: str) -> ModelFile | None:
    if not name:
        return None
    lower = name.replace("/", "\\").lower()
    normalized_lower = _normalize_model_lookup(lower)
    for items in by_category.values():
        for item in items:
            item_name = item.name.replace("/", "\\").lower()
            item_relative = item.relative_path.replace("/", "\\").lower()
            item_lookup = _normalize_model_lookup(item_relative)
            if item_name == lower or item_relative == lower or item_lookup == normalized_lower:
                return item
    return None


def _normalize_model_lookup(value: str) -> str:
    text = value.strip().replace("/", "\\")
    for prefix in ("models\\", ".\\models\\"):
        if text.startswith(prefix):
            text = text[len(prefix) :]
    if text.startswith("loras\\ltx2\\"):
        text = "loras\\ltx\\" + text.split("\\", 2)[2]
    if text.startswith("ltx2\\"):
        text = "ltx\\" + text.split("\\", 1)[1]
    return text


def _is_wan_noise_model(item: ModelFile, noise: str) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "wan" in haystack and noise.lower() in haystack


def _selected_asset(value: object) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "automatic", "auto", "none"} else text


def _first(
    by_category: dict[str, list[ModelFile]],
    categories: list[str],
    tokens: list[str],
    *,
    include_extensions: set[str] | None = None,
    exclude_extensions: set[str] | None = None,
) -> ModelFile | None:
    lowered = [token.lower() for token in tokens]
    for category in categories:
        for item in by_category.get(category, []):
            if include_extensions and item.extension.lower() not in include_extensions:
                continue
            if exclude_extensions and item.extension.lower() in exclude_extensions:
                continue
            haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
            if all(token in haystack for token in lowered):
                return item
    if not tokens:
        for category in categories:
            if by_category.get(category):
                return by_category[category][0]
    return None
