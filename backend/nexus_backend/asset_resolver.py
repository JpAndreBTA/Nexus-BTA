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
    selected_requests_flux2 = preset == "flux" and any(
        token in selected_name.lower() for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein")
    )
    if selected_requests_flux2 and not selected_model:
        assets.pop("primary_model", None)
    if preset == "ltx":
        assets.update(_resolve_ltx(by_category, selected_name, request))
    elif preset == "wan":
        assets.update(_resolve_wan(by_category, selected_name, request))
    elif preset == "flux":
        assets.update(_resolve_flux(by_category, selected_name, request))
    elif preset == "anima":
        assets.update(_resolve_anima(by_category, selected_name, request))
    elif preset in {"ideogram4", "ideogram"}:
        assets.update(_resolve_ideogram4(by_category, selected_name, request))
    elif preset == "qwen":
        assets.pop("primary_model", None)
        assets.update(_resolve_qwen(by_category, selected_name, request))
    elif preset in {"zimageturbo", "zimage"}:
        assets.update(_resolve_zimage(by_category, selected_name, request))
    elif preset in {"sd", "sd15"}:
        assets.update(_resolve_sd_family(by_category, selected_name, request, [["sd15"], ["sd", "1.5"], ["realistic"]]))
    elif preset in {"xl", "sdxl"}:
        assets.update(_resolve_sd_family(by_category, selected_name, request, [["sdxl"], ["xl"], ["illustrious"]]))
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
    if preset == "ltx":
        selected = _selected_model_choice(by_category, control.model)
        if selected and selected.category != "loras":
            selected = None
        model = selected or _first_ltx_lora(by_category, [["ic-lora", "union"], ["ic-lora", "control"], ["ic-lora", "cameraman"], ["ic", "lora"]])
        if not model:
            return {}
        return {"controlnet_model": _comfy_name(model), "ic_lora": _comfy_name(model)}
    if preset not in {"sd", "sd15", "xl", "sdxl", "flux", "qwen", "zimageturbo", "zimage"}:
        return {}
    selected_name = Path(request.model_path or request.model_name or "").name
    selected = _selected_model_choice(by_category, control.model)
    selected_model_haystack = " ".join([selected_name, request.model_name or "", request.model_path or ""]).lower()
    flux2_route = preset == "flux" and any(
        token in selected_model_haystack
        for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein")
    )
    controlnet_categories = {"controlnet", "model_patches"}
    if selected and selected.category not in controlnet_categories:
        selected = None
    control_type = str(control.type or "").lower()
    candidates = [*by_category.get("controlnet", []), *by_category.get("model_patches", [])]
    if flux2_route:
        flux2_candidates = [
            item
            for item in candidates
            if any(
                token in " ".join([item.name, item.folder, item.relative_path]).lower()
                for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein")
            )
        ]
        candidates = flux2_candidates
        if selected and selected not in candidates:
            selected = None
    if selected and not _controlnet_matches_preset_type(selected, preset, control_type):
        selected = None
    model = selected
    if not model:
        model = _first_controlnet(candidates, preset, control_type)
    if not model:
        return {}
    return {"controlnet_model": _comfy_name(model), "controlnet_category": model.category}


def _resolve_sd_family(
    by_category: dict[str, list[ModelFile]],
    selected_name: str,
    request: GenerateRequest,
    token_sets: list[list[str]],
) -> dict[str, str]:
    assets: dict[str, str] = {}
    primary = _find_name(by_category, selected_name)
    if primary and primary.category not in {"checkpoints", "diffusion_models", "unet"}:
        primary = None
    for tokens in token_sets:
        if primary:
            break
        primary = _first(by_category, ["checkpoints", "diffusion_models", "unet"], tokens)
    if primary:
        assets["primary_model"] = _comfy_name(primary)
    selected_vae = _selected_model_choice(by_category, request.vae)
    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    if selected_vae and selected_vae.category == "vae":
        assets["vae"] = _comfy_name(selected_vae)
    if selected_text_encoder and selected_text_encoder.category in {"text_encoders", "clip"}:
        assets["text_encoder"] = _comfy_name(selected_text_encoder)
    return assets


def _controlnet_matches_preset_type(item: ModelFile, preset: str, control_type: str) -> bool:
    if preset in {"xl", "sdxl"}:
        preset_tokens = ["sdxl", "xl"]
    elif preset == "qwen":
        preset_tokens = ["qwen", "qwen-image"]
    elif preset in {"zimageturbo", "zimage"}:
        preset_tokens = ["z-image", "zimage", "z_image"]
    elif preset == "flux":
        preset_tokens = ["flux", "flux.1", "flux1"]
    else:
        preset_tokens = ["sd15", "sd1", "v11", "1.5"]
    type_tokens = {
        "dwpose": ["dwpose", "dw pose", "openpose", "pose"],
        "openpose": ["openpose", "pose"],
        "pose": ["openpose", "pose"],
        "depth": ["depth"],
        "canny": ["canny"],
        "lineart": ["lineart", "line"],
        "tile": ["tile"],
        "softedge": ["softedge", "soft", "hed"],
        "normal": ["normal"],
    }.get(control_type, [control_type] if control_type else [])
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    if preset == "qwen" and ("qwen" not in haystack and "instantx" not in haystack and "diffsynth" not in haystack):
        return False
    if preset in {"zimageturbo", "zimage"} and not any(token in haystack for token in preset_tokens + ["fun", "controlnet-union"]):
        return False
    if preset not in {"qwen", "zimageturbo", "zimage"} and not any(token in haystack for token in preset_tokens):
        return False
    union_model = (
        preset == "qwen" and ("union" in haystack or "instantx" in haystack)
    ) or (
        preset in {"zimageturbo", "zimage"} and ("union" in haystack or "fun" in haystack)
    ) or (
        preset == "flux" and ("union" in haystack or "shakker" in haystack)
    )
    if type_tokens and not union_model and not any(token in haystack for token in type_tokens):
        return False
    return True


def _first_controlnet(items: list[ModelFile], preset: str, control_type: str) -> ModelFile | None:
    scored: list[tuple[int, str, ModelFile]] = []
    for item in items:
        if not _controlnet_matches_preset_type(item, preset, control_type):
            continue
        scored.append((len(item.name), item.name.lower(), item))
    if not scored:
        return None
    scored.sort(key=lambda row: row[:2])
    return scored[0][2]


def _resolve_flux(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    primary = _find_name(by_category, selected_name)
    selected_requests_flux2 = any(token in str(selected_name or "").lower() for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein"))
    if not primary and not selected_requests_flux2:
        primary = (
            _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["flux", "q5"])
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["flux", "dev"])
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["flux"])
        )
    if primary:
        assets["primary_model"] = _comfy_name(primary)

    family = _flux_family(primary, selected_name)
    assets["flux_family"] = family
    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    selected_vae = _selected_model_choice(by_category, request.vae)
    if family.startswith("flux2"):
        text_encoder = _resolve_flux2_text_encoder(by_category, selected_text_encoder, family)
        vae = _resolve_flux2_vae(by_category, selected_vae)
        if text_encoder:
            assets["text_encoder"] = _comfy_name(text_encoder)
        if vae:
            assets["vae"] = _comfy_name(vae)
        if _truthy((request.video or {}).get("flux_multiview")):
            multiangle_lora = _first_flux_multiangle_lora(by_category)
            if multiangle_lora:
                assets["flux_multiangle_lora"] = _comfy_name(multiangle_lora)
        return assets

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


def _flux_family(model: ModelFile | None, selected_name: str = "") -> str:
    haystack = " ".join(
        value
        for value in [
            selected_name,
            model.name if model else "",
            model.relative_path if model else "",
            model.folder if model else "",
        ]
        if value
    ).lower()
    if any(token in haystack for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein")):
        if "klein" in haystack:
            if "9b" in haystack:
                return "flux2_klein_9b"
            return "flux2_klein_4b"
        return "flux2_dev"
    return "flux1"


def _resolve_flux2_text_encoder(
    by_category: dict[str, list[ModelFile]],
    selected: ModelFile | None,
    family: str,
) -> ModelFile | None:
    if selected:
        selected_haystack = " ".join([selected.name, selected.folder, selected.relative_path]).lower()
        if family == "flux2_dev" and "mistral" in selected_haystack:
            return selected
        if family == "flux2_klein_9b" and any(token in selected_haystack for token in ("qwen_3_8b", "qwen3_8b", "8b")):
            return selected
        if family == "flux2_klein_4b" and any(token in selected_haystack for token in ("qwen_3_4b", "qwen3_4b", "4b")):
            return selected
        return selected
    if family == "flux2_dev":
        return (
            _first(by_category, ["text_encoders", "clip"], ["mistral", "flux2"])
            or _first(by_category, ["text_encoders", "clip"], ["mistral_3_small"])
            or _first(by_category, ["text_encoders", "clip"], ["mistral"])
        )
    if family == "flux2_klein_9b":
        return (
            _first(by_category, ["text_encoders", "clip"], ["qwen_3_8b"])
            or _first(by_category, ["text_encoders", "clip"], ["qwen3", "8b"])
            or _first(by_category, ["text_encoders", "clip"], ["qwen", "8b"])
        )
    return (
        _first(by_category, ["text_encoders", "clip"], ["qwen_3_4b"])
        or _first(by_category, ["text_encoders", "clip"], ["qwen3", "4b"])
        or _first(by_category, ["text_encoders", "clip"], ["qwen", "4b"])
    )


def _resolve_flux2_vae(by_category: dict[str, list[ModelFile]], selected: ModelFile | None) -> ModelFile | None:
    if selected:
        haystack = " ".join([selected.name, selected.folder, selected.relative_path]).lower()
        if "flux2" in haystack or "flux-2" in haystack or "full_encoder" in haystack:
            return selected
        return selected
    return (
        _first(by_category, ["vae"], ["flux2", "vae"])
        or _first(by_category, ["vae"], ["flux2-vae"])
        or _first(by_category, ["vae"], ["full_encoder", "decoder"])
    )


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
    raw_latent_upscale = str(video_options.get("latent_upscale") or "").strip().lower()
    latent_upscale_disabled = raw_latent_upscale in {"none", "off", "disabled", "false", "0", "no"}
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

    upscale = None if latent_upscale_disabled else (
        _find_name(by_category, selected_upscale)
        or _first(by_category, ["latent_upscale_models", "upscale_models"], ["ltx-2.3", "spatial", "x2"])
        or _first(by_category, ["latent_upscale_models", "upscale_models"], ["spatial", "x2"])
        or _first(by_category, ["latent_upscale_models", "upscale_models"], ["ltx", "spatial"])
    )
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

    selected_control_lora = _selected_model_choice(by_category, getattr(request.controlnet, "model", None))
    if selected_control_lora and selected_control_lora.category != "loras":
        selected_control_lora = None
    ic_lora = (
        selected_control_lora
        if selected_control_lora and "ic" in " ".join([selected_control_lora.name, selected_control_lora.folder, selected_control_lora.relative_path]).lower()
        else None
    ) or _first_ltx_lora(by_category, [["ltx_ic", "union"], ["ic-lora", "union"], ["union", "control"], ["ltx_ic", "control"], ["ic-lora", "control"]])
    if ic_lora:
        assets["ic_lora"] = _comfy_name(ic_lora)
    cameraman_lora = _first_ltx_lora(by_category, [["cameraman"], ["camera", "motion"]])
    if cameraman_lora:
        assets["cameraman_lora"] = _comfy_name(cameraman_lora)
    detailer_lora = _first_ltx_lora(by_category, [["detailer"], ["ic-lora", "detail"]])
    if detailer_lora:
        assets["detailer_lora"] = _comfy_name(detailer_lora)
    outpaint_lora = _first_ltx_lora(by_category, [["outpaint"], ["outpainting"]])
    if outpaint_lora:
        assets["outpaint_lora"] = _comfy_name(outpaint_lora)
    id_lora = _first_ltx_lora(by_category, [["id-lora"], ["id", "lora"], ["celebvhq"]])
    if id_lora:
        assets["id_lora"] = _comfy_name(id_lora)
    transition_lora = _first_ltx_lora(by_category, [["transition"], ["zhuanchang"]])
    if transition_lora:
        assets["transition_lora"] = _comfy_name(transition_lora)
    return assets


def _resolve_wan(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    selected = _find_name(by_category, selected_name)
    selected_base = selected if selected and _is_wan_base_model(selected) else None
    high_model = selected_base if selected_base and _is_wan_noise_model(selected_base, "high") else None
    low_model = selected_base if selected_base and _is_wan_noise_model(selected_base, "low") else None

    high_model = high_model or (
        _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan2.2", "high"])
        or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan", "high"])
    )
    low_model = low_model or (
        _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["dasiwa", "low"])
        or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["tasty", "low"])
        or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan2.2", "low"])
        or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["wan", "low"])
    )

    single_file_model = selected_base if selected_base and _is_wan_single_file_model(selected_base) else None
    if single_file_model and _is_wan_fun_control_single_model(single_file_model):
        high_model = single_file_model
        low_model = single_file_model
    elif single_file_model and (not high_model or not low_model):
        high_model = high_model or single_file_model
        low_model = low_model or single_file_model

    primary = selected_base or high_model or low_model
    if primary:
        assets["primary_model"] = _comfy_name(primary)
    if high_model:
        assets["wan_high_model"] = _comfy_name(high_model)
    if low_model:
        assets["wan_low_model"] = _comfy_name(low_model)

    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    selected_vae = _selected_model_choice(by_category, request.vae)
    model_haystack = " ".join(
        value
        for value in [
            selected_name,
            high_model.name if high_model else "",
            low_model.name if low_model else "",
            high_model.relative_path if high_model else "",
            low_model.relative_path if low_model else "",
        ]
        if value
    ).lower()
    wan_ti2v_route = "ti2v" in model_haystack or "5b" in model_haystack
    wan22_route = any(token in model_haystack for token in ("wan2.2", "wan22", "wan_2.2"))
    text_encoder = (
        selected_text_encoder
        or
        _first(by_category, ["text_encoders", "clip"], ["umt5"])
        or _first(by_category, ["text_encoders", "clip"], ["t5"])
    )
    vae = selected_vae or (
        (_first(by_category, ["vae"], ["wan22"]) or _first(by_category, ["vae"], ["wan2.2"]) if wan_ti2v_route else None)
        or (_first(by_category, ["vae"], ["wan_2.1"]) or _first(by_category, ["vae"], ["wan2.1"]))
        or _first(by_category, ["vae"], ["wan"])
    )
    if text_encoder:
        assets["text_encoder"] = _comfy_name(text_encoder)
    if vae:
        assets["vae"] = _comfy_name(vae)
    clip_vision = (
        _first(by_category, ["clip_vision"], ["clip_vision"])
        or _first(by_category, ["clip_vision"], ["siglip"])
        or _first(by_category, ["clip_vision"], ["vision"])
        or _first(by_category, ["clip_vision"], [])
    )
    if clip_vision:
        assets["clip_vision"] = _comfy_name(clip_vision)
    high_lora = _first_wan_4step_lora(by_category, "high")
    low_lora = _first_wan_4step_lora(by_category, "low")
    if high_lora and low_lora:
        assets["wan_4step_high_lora"] = _comfy_name(high_lora)
        assets["wan_4step_low_lora"] = _comfy_name(low_lora)
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


def _resolve_ideogram4(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    primary = _find_name(by_category, selected_name)
    if primary and not _is_ideogram4_model(primary):
        primary = None
    if not primary:
        primary = (
            _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["ideogram4", "fp8"])
            or _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["ideogram", "4"])
        )
    if primary:
        assets["primary_model"] = _comfy_name(primary)

    unconditional = (
        _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["ideogram4", "unconditional"])
        or _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["ideogram", "unconditional"])
    )
    if unconditional:
        assets["ideogram4_unconditional_model"] = _comfy_name(unconditional)

    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    text_encoder = (
        selected_text_encoder
        or _first_exact(by_category, ["text_encoders", "clip"], "qwen3vl_8b_fp8_scaled.safetensors")
        or _first(by_category, ["text_encoders", "clip"], ["qwen3vl"])
        or _first(by_category, ["text_encoders", "clip"], ["qwen3", "vl"])
    )
    if text_encoder:
        assets["text_encoder"] = _comfy_name(text_encoder)

    selected_vae = _selected_model_choice(by_category, request.vae)
    vae = selected_vae or _first_exact(by_category, ["vae"], "flux2-vae.safetensors") or _first(by_category, ["vae"], ["flux2", "vae"])
    if vae:
        assets["vae"] = _comfy_name(vae)

    gemma = (
        _first_exact(by_category, ["text_encoders", "clip"], "gemma4_e4b_it_fp8_scaled.safetensors")
        or _first(by_category, ["text_encoders", "clip"], ["gemma4"])
    )
    if gemma:
        assets["ideogram4_gemma_prompt_encoder"] = _comfy_name(gemma)
    return assets


def _resolve_qwen(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    has_reference = request.activity == "img2img" and (
        bool((request.img2img.reference_image or "").strip())
        or any(bool((value or "").strip()) for value in (request.img2img.reference_images or []))
    )
    primary = _find_name(by_category, selected_name)
    raw_selected_edit = selected_name if has_reference and _looks_like_qwen_edit_name(selected_name) else ""
    if has_reference and primary and not _is_qwen_edit_model(primary):
        preferred_edit = (
            _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["qwen", "edit", "q4"])
            or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["qwen", "edit"])
        )
        if preferred_edit:
            primary = preferred_edit
    if primary and _is_qwen_edit_model(primary) and not has_reference:
        primary = None
    if raw_selected_edit and not primary:
        assets["primary_model"] = raw_selected_edit
    if not primary:
        if has_reference and not raw_selected_edit:
            primary = (
                _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["qwen", "edit", "q4"])
                or _first(by_category, ["checkpoints", "unet", "diffusion_models"], ["qwen", "edit"])
            )
        if not raw_selected_edit:
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
    if has_reference:
        edit_lightning = _first_qwen_edit_lightning_lora(
            by_category,
            _comfy_name(primary) if primary else (raw_selected_edit or selected_name),
        )
        if edit_lightning:
            assets["qwen_edit_lightning_lora"] = _comfy_name(edit_lightning)
    if has_reference and _truthy((request.video or {}).get("qwen_multiview")):
        multiangle_lora = _first_qwen_multiangle_lora(by_category)
        if multiangle_lora:
            assets["qwen_multiangle_lora"] = _comfy_name(multiangle_lora)
    return assets


def _resolve_zimage(by_category: dict[str, list[ModelFile]], selected_name: str, request: GenerateRequest) -> dict[str, str]:
    assets: dict[str, str] = {}
    primary = _find_name(by_category, selected_name)
    if primary and primary.category in {"vae", "text_encoders", "clip", "loras", "embeddings"}:
        primary = None
    if not primary:
        primary = (
            _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["z", "image", "turbo"])
            or _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["zimage", "turbo"])
            or _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["z-image", "turbo"])
            or _first(by_category, ["diffusion_models", "unet", "checkpoints"], ["z", "image"])
        )
    if primary:
        assets["primary_model"] = _comfy_name(primary)

    selected_text_encoder = _selected_model_choice(by_category, request.text_encoder)
    text_encoder = selected_text_encoder or _first_exact(by_category, ["text_encoders", "clip"], "qwen_3_4b.safetensors")
    if text_encoder:
        assets["text_encoder"] = _comfy_name(text_encoder)

    selected_vae = _selected_model_choice(by_category, request.vae)
    vae = selected_vae or _first_exact(by_category, ["vae"], "ae.safetensors")
    if vae:
        assets["vae"] = _comfy_name(vae)
    return assets


def _selected_model_choice(by_category: dict[str, list[ModelFile]], value: str | None) -> ModelFile | None:
    raw = str(value or "").strip()
    if not raw or raw.lower() in {"automatic", "auto", "none"}:
        return None
    return _find_name(by_category, raw) or _find_name(by_category, Path(raw).name)


def _is_exact_model_name(item: ModelFile, name: str) -> bool:
    return Path(item.name).name.lower() == name.lower()


def _is_ideogram4_model(item: ModelFile) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "ideogram" in haystack and "4" in haystack and "unconditional" not in haystack


def _is_ideogram4_qwen3vl_encoder(item: ModelFile) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "qwen3vl" in haystack or ("qwen3" in haystack and "vl" in haystack)


def _first_exact(by_category: dict[str, list[ModelFile]], categories: list[str], name: str) -> ModelFile | None:
    expected = name.lower()
    for category in categories:
        for item in by_category.get(category, []):
            if Path(item.name).name.lower() == expected:
                return item
    return None


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


def _looks_like_qwen_edit_name(value: str | None) -> bool:
    haystack = str(value or "").lower()
    return "qwen" in haystack and "edit" in haystack and any(
        haystack.endswith(extension) for extension in (".gguf", ".safetensors", ".sft")
    )


def _first_qwen_edit_lightning_lora(by_category: dict[str, list[ModelFile]], model_name: str | None = None) -> ModelFile | None:
    preferred: list[tuple[int, ModelFile]] = []
    model_text = str(model_name or "").lower()
    target_version = "2511" if "2511" in model_text else ("2509" if "2509" in model_text else "")
    for item in by_category.get("loras", []):
        haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
        if "qwen" not in haystack or "lightning" not in haystack:
            continue
        score = 0
        if "edit" in haystack:
            score += 50
        if target_version and target_version in haystack:
            score += 100
        elif target_version and any(version in haystack for version in ("2509", "2511")):
            score -= 80
        if "2511" in haystack:
            score += 40
        if "4steps" in haystack or "4step" in haystack:
            score += 20
        if "bf16" in haystack:
            score += 8
        if "fp32" in haystack:
            score += 5
        if "2509" in haystack:
            score += 10
        preferred.append((score, item))
    if not preferred:
        return None
    preferred.sort(key=lambda pair: pair[0], reverse=True)
    return preferred[0][1]


def _first_qwen_multiangle_lora(by_category: dict[str, list[ModelFile]]) -> ModelFile | None:
    preferred: list[tuple[int, ModelFile]] = []
    for item in by_category.get("loras", []):
        haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
        if "qwen" not in haystack:
            continue
        if not any(token in haystack for token in ("multiangle", "multi-angle", "multiple-angle", "multiple-angles", "angles-lora")):
            continue
        score = 0
        if "2511" in haystack:
            score += 40
        if "edit" in haystack:
            score += 20
        if "fal" in haystack:
            score += 5
        preferred.append((score, item))
    if not preferred:
        return None
    preferred.sort(key=lambda pair: pair[0], reverse=True)
    return preferred[0][1]


def _first_flux_multiangle_lora(by_category: dict[str, list[ModelFile]]) -> ModelFile | None:
    preferred: list[tuple[int, ModelFile]] = []
    for item in by_category.get("loras", []):
        haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
        if "flux" not in haystack:
            continue
        if not any(token in haystack for token in ("multiangle", "multi-angle", "multiple-angle", "multiple-angles", "angles-flux")):
            continue
        score = 0
        if "klein" in haystack:
            score += 40
        if "9b" in haystack:
            score += 10
        preferred.append((score, item))
    if not preferred:
        return None
    preferred.sort(key=lambda pair: pair[0], reverse=True)
    return preferred[0][1]


def _truthy(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "off", "none", "no"}
    return bool(value)


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
        if item.category == "loras" and name.lower().startswith("ltx_ic\\"):
            return "ltx_ic\\" + name.split("\\", 1)[1]
        return name
    if relative:
        if relative.lower().startswith("ltx2\\"):
            return "ltx\\" + relative.split("\\", 1)[1]
        return relative
    return item.name


def _find_name(by_category: dict[str, list[ModelFile]], name: str) -> ModelFile | None:
    if not name:
        return None
    lower = name.replace("/", "\\").lower()
    normalized_lower = _normalize_model_lookup(lower)
    lookup_variants = {normalized_lower}
    if "\\" not in normalized_lower and by_category.get("loras"):
        lookup_variants.update(
            {
                f"loras\\{normalized_lower}",
                f"loras\\ltx\\{normalized_lower}",
                f"loras\\ltx_ic\\{normalized_lower}",
            }
        )
    elif normalized_lower.startswith("ltx\\") or normalized_lower.startswith("ltx_ic\\"):
        lookup_variants.add(f"loras\\{normalized_lower}")
    elif normalized_lower.startswith("loras\\ltx\\"):
        lookup_variants.add("loras\\ltx_ic\\" + normalized_lower.split("\\", 2)[2])
    elif normalized_lower.startswith("loras\\ltx_ic\\"):
        lookup_variants.add("loras\\ltx\\" + normalized_lower.split("\\", 2)[2])
    for items in by_category.values():
        for item in items:
            item_name = item.name.replace("/", "\\").lower()
            item_relative = item.relative_path.replace("/", "\\").lower()
            item_path = item.path.replace("/", "\\").lower()
            item_lookup = _normalize_model_lookup(item_relative)
            item_path_lookup = _normalize_model_lookup(item_path)
            if item_name == lower or item_relative == lower or item_path == lower or item_lookup in lookup_variants or item_path_lookup in lookup_variants:
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
    if text.startswith("loras\\") and text.count("\\") == 1:
        filename = text.split("\\", 1)[1]
        if any(token in filename for token in ("ic-lora", "ic_lora", "cameraman", "detailer", "union-control", "union_control")):
            text = "loras\\ltx_ic\\" + filename
    if text.startswith("ltx\\"):
        filename = text.split("\\", 1)[1]
        if any(token in filename for token in ("ic-lora", "ic_lora", "cameraman", "detailer", "union-control", "union_control")):
            text = "ltx_ic\\" + filename
    return text


def _is_wan_noise_model(item: ModelFile, noise: str) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return "wan" in haystack and noise.lower() in haystack


def _is_wan_base_model(item: ModelFile) -> bool:
    return item.category in {"checkpoints", "unet", "diffusion_models"} and "wan" in " ".join(
        [item.name, item.folder, item.relative_path]
    ).lower()


def _is_wan_single_file_model(item: ModelFile) -> bool:
    if not _is_wan_base_model(item) or item.extension.lower() != ".gguf":
        return False
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return not any(token in haystack for token in ("animate-lora", "motion-adapter", "motion_adapter"))


def _is_wan_fun_control_single_model(item: ModelFile) -> bool:
    haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
    return item.extension.lower() == ".gguf" and "fun" in haystack and "control" in haystack


def _first_wan_4step_lora(by_category: dict[str, list[ModelFile]], noise: str) -> ModelFile | None:
    noise = noise.lower()
    candidates: list[tuple[int, str, ModelFile]] = []
    for item in by_category.get("loras", []):
        haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
        if "wan" not in haystack or noise not in haystack:
            continue
        if not any(token in haystack for token in ("lightx2v", "4step", "4-step", "lightning", "distill")):
            continue
        score = 0
        if "lightx2v" in haystack:
            score -= 40
        if "4step" in haystack or "4-step" in haystack:
            score -= 30
        if "i2v" in haystack:
            score -= 10
        candidates.append((score, item.name.lower(), item))
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[:2])
    return candidates[0][2]


def _first_ltx_lora(by_category: dict[str, list[ModelFile]], token_sets: list[list[str]]) -> ModelFile | None:
    candidates: list[tuple[int, str, ModelFile]] = []
    for item in by_category.get("loras", []):
        haystack = " ".join([item.name, item.folder, item.relative_path]).lower()
        if "ltx" not in haystack:
            continue
        for rank, tokens in enumerate(token_sets):
            if all(token.lower() in haystack for token in tokens):
                candidates.append((rank, len(item.name), item.name.lower(), item))
                break
    if not candidates:
        return None
    candidates.sort(key=lambda row: row[:3])
    return candidates[0][3]


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
