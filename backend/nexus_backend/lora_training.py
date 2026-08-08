from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import NexusSettings, runtime_python


TRAIN_LORA_SOURCES = [
    {
        "label": "ComfyUI LTX multi-frame workflows",
        "url": "https://docs.comfy.org/tutorials/video/ltxv",
        "note": "Official ComfyUI LTX examples include multi-frame control with start and end frames.",
    },
    {
        "label": "LTX-2 Trainer",
        "url": "https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer",
        "note": "Official LTX trainer package for LoRA, full fine-tuning and IC-LoRA.",
    },
    {
        "label": "LTX-2 Training Docs",
        "url": "https://docs.ltx.video/open-source-model/ltx-2-trainer/ltx-2-training",
        "note": "LTX-2.3 training needs a local checkpoint, Gemma text encoder, CUDA Linux runtime and high VRAM for full runs.",
    },
    {
        "label": "LTX IC-LoRA Docs",
        "url": "https://docs.ltx.video/open-source-model/usage-guides/ic-lo-ra",
        "note": "IC-LoRA uses paired video/control signals such as depth, pose, edges or sparse tracks.",
    },
    {
        "label": "RunComfy LTX 2.3 LoRA baseline",
        "url": "https://www.runcomfy.com/trainer/ai-toolkit/ltx-2-3-lora-training-guide",
        "note": "Practical LTX 2.3 LoRA baseline: rank 32, LR 1e-4, 3000 steps, 512/768/1024 datasets and fixed validation samples.",
    },
    {
        "label": "kohya-ss sd-scripts",
        "url": "https://github.com/kohya-ss/sd-scripts",
        "note": "Common SD 1.5/SDXL LoRA runner via train_network.py and sdxl_train_network.py.",
    },
    {
        "label": "AI Toolkit",
        "url": "https://github.com/ostris/ai-toolkit",
        "note": "Diffusion training suite with FLUX and Qwen Image/Edit support.",
    },
    {
        "label": "AI Toolkit Perceptual",
        "url": "https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual",
        "note": "AI Toolkit fork with depth/identity/body perceptual anchors and weight noising for small or single-image LoRAs.",
    },
]


PERCEPTUAL_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "anchor": "depth",
    "image_mode": "auto",
    "depth_loss_weight": 0.1,
    "depth_model_id": "depth-anything/Depth-Anything-V2-Small-hf",
    "depth_mask_source": "subject",
    "loss_min_t": 0.0,
    "loss_max_t": 1.0,
    "preview_every": 100,
    "loss_split": "auto",
    "weight_noise_enabled": True,
    "weight_noise_mode": "relative",
    "weight_noise_sigma": 0.0125,
    "weight_noise_log_every": 50,
    "single_image_sigma": 0.04,
    "multi_image_sigma": 0.012,
}


TRAIN_LORA_PRESETS: dict[str, dict[str, Any]] = {
    "SD": {
        "label": "SD 1.5",
        "base": "Stable Diffusion 1.5",
        "trainer": "kohya_ss",
        "trainer_label": "kohya-ss sd-scripts",
        "script": "train_network.py",
        "resolution": 512,
        "caption": "Use a short trigger token plus concrete subject/style captions.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 1e-4, "steps": 1800, "repeats": 12, "batch_size": 2},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 8e-5, "steps": 1200, "repeats": 8, "batch_size": 2},
        },
    },
    "XL": {
        "label": "SDXL",
        "base": "Stable Diffusion XL",
        "trainer": "kohya_ss",
        "trainer_label": "kohya-ss sd-scripts",
        "script": "sdxl_train_network.py",
        "resolution": 1024,
        "caption": "Prefer 1024px buckets, captions with trigger token and visual attributes.",
        "perceptual": {
            "compatible": True,
            "support_level": "supported",
            "recommended_for": ["character", "style"],
            "notes": "Use the perceptual AI Toolkit fork for SDXL depth anchors or small/single-image regularization.",
        },
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 8e-5, "steps": 2200, "repeats": 10, "batch_size": 1},
            "style": {"rank": 12, "alpha": 12, "learning_rate": 6e-5, "steps": 1600, "repeats": 8, "batch_size": 1},
        },
    },
    "Illustrious": {
        "label": "Illustrious",
        "base": "Illustrious XL",
        "trainer": "kohya_ss",
        "trainer_label": "kohya-ss sd-scripts",
        "script": "sdxl_train_network.py",
        "resolution": 1024,
        "caption": "Use Danbooru-style tags with a stable trigger; keep quality tags out of captions when possible.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 8e-5, "steps": 2200, "repeats": 10, "batch_size": 1},
            "style": {"rank": 12, "alpha": 12, "learning_rate": 6e-5, "steps": 1600, "repeats": 8, "batch_size": 1},
        },
    },
    "Pony": {
        "label": "Pony",
        "base": "Pony Diffusion XL",
        "trainer": "kohya_ss",
        "trainer_label": "kohya-ss sd-scripts",
        "script": "sdxl_train_network.py",
        "resolution": 1024,
        "caption": "Use Pony score/source tags consistently and keep the trigger near the start.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 8e-5, "steps": 2200, "repeats": 10, "batch_size": 1},
            "style": {"rank": 12, "alpha": 12, "learning_rate": 6e-5, "steps": 1600, "repeats": 8, "batch_size": 1},
        },
    },
    "Flux": {
        "label": "FLUX",
        "base": "FLUX.1",
        "trainer": "ai_toolkit",
        "trainer_label": "AI Toolkit",
        "resolution": 1024,
        "caption": "Use natural captions; keep trigger token rare and consistent.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 1e-4, "steps": 1800, "repeats": 8, "batch_size": 1},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 8e-5, "steps": 1200, "repeats": 6, "batch_size": 1},
        },
    },
    "Flux2": {
        "label": "FLUX 2",
        "base": "FLUX.2",
        "trainer": "ai_toolkit",
        "trainer_label": "AI Toolkit",
        "resolution": 1024,
        "caption": "Use natural language captions and a rare trigger token; keep batches small on consumer GPUs.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 8e-5, "steps": 1800, "repeats": 8, "batch_size": 1},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 6e-5, "steps": 1200, "repeats": 6, "batch_size": 1},
        },
    },
    "Flux2Klein": {
        "label": "FLUX 2 Klein",
        "base": "FLUX.2 Klein",
        "trainer": "ai_toolkit",
        "trainer_label": "AI Toolkit",
        "resolution": 1024,
        "caption": "Klein presets favor low rank and cached latents for fast iteration.",
        "perceptual": {
            "compatible": True,
            "support_level": "supported",
            "recommended_for": ["character", "style"],
            "notes": "Best-documented target for weight noising and depth anchors in the perceptual fork.",
        },
        "templates": {
            "character": {"rank": 8, "alpha": 8, "learning_rate": 9e-5, "steps": 1400, "repeats": 8, "batch_size": 1},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 7e-5, "steps": 1000, "repeats": 6, "batch_size": 1},
        },
    },
    "Qwen": {
        "label": "QWEN",
        "base": "Qwen Image/Edit",
        "trainer": "ai_toolkit",
        "trainer_label": "AI Toolkit",
        "resolution": 1024,
        "caption": "Caption edit intent, retained identity, and the visual delta from the source.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 7e-5, "steps": 2000, "repeats": 8, "batch_size": 1},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 6e-5, "steps": 1400, "repeats": 6, "batch_size": 1},
        },
    },
    "ZImageTurbo": {
        "label": "Z-IMG",
        "base": "Z-Image Turbo",
        "trainer": "simpletuner",
        "trainer_label": "SimpleTuner compatible",
        "resolution": 1024,
        "caption": "Use clean image captions and conservative learning rate for turbo checkpoints.",
        "perceptual": {
            "compatible": True,
            "support_level": "experimental",
            "recommended_for": ["character"],
            "notes": "The perceptual fork lists Z-Image Turbo as experimental; keep depth and weight-noise settings conservative.",
        },
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 5e-5, "steps": 1800, "repeats": 8, "batch_size": 1},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 4e-5, "steps": 1200, "repeats": 6, "batch_size": 1},
        },
    },
    "Lumina": {
        "label": "LUMINA",
        "base": "Lumina Image",
        "trainer": "simpletuner",
        "trainer_label": "SimpleTuner compatible",
        "resolution": 1024,
        "caption": "Prefer varied aesthetic captions and lower LR for style transfer.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 6e-5, "steps": 1800, "repeats": 8, "batch_size": 1},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 5e-5, "steps": 1200, "repeats": 6, "batch_size": 1},
        },
    },
    "Wan": {
        "label": "WAN",
        "base": "Wan 2.2 Video",
        "trainer": "musubi_tuner",
        "trainer_label": "musubi-tuner / Wan trainer",
        "resolution": 832,
        "frames": 81,
        "caption": "Use short clips with motion captions; cache latents before long runs.",
        "templates": {
            "character": {"rank": 32, "alpha": 16, "learning_rate": 2e-5, "steps": 2400, "repeats": 4, "batch_size": 1},
            "style": {"rank": 16, "alpha": 8, "learning_rate": 2e-5, "steps": 1800, "repeats": 4, "batch_size": 1},
        },
    },
    "LTX": {
        "label": "LTX 2.3",
        "base": "LTX 2.3 Video",
        "trainer": "ltx_trainer",
        "trainer_label": "LTX-2 trainer",
        "resolution": "512+768+1024",
        "frames": 1,
        "caption": "Use RunComfy's LTX-2.3 baseline: T2V first, rank 32, LR 1e-4, 3000 steps, 512+768+1024 datasets and fixed validation. Character/style can start with frames=1; IC-LoRA needs paired target/control signals.",
        "perceptual": {
            "compatible": True,
            "support_level": "experimental",
            "recommended_for": ["character", "style", "motion"],
            "notes": "The perceptual fork lists LTX-2.3 video as experimental, including depth consistency across frames.",
        },
        "templates": {
            "character": {
                "mode_label": "Character LoRA",
                "training_mode": "lora",
                "rank": 32,
                "alpha": 32,
                "learning_rate": 1e-4,
                "steps": 3000,
                "repeats": 1,
                "batch_size": 1,
                "frames": 1,
                "dataset_mode": "clip_caption",
                "dataset_resolutions": [512, 768, 1024],
                "control_type": "none",
                "attention_strength": 1.0,
                "target_fps": 24,
                "caption_dropout": 0.05,
                "weight_decay": 0.0001,
                "timestep_type": "weighted",
                "timestep_bias": "balanced",
                "loss_type": "mse",
                "sample_every": 250,
                "sample_sampler": "flowmatch",
                "sample_guidance_scale": 4,
                "sample_steps": 25,
                "sample_size": "768x768",
                "cache_text_embeddings": False,
            },
            "style": {
                "mode_label": "Style LoRA",
                "training_mode": "lora",
                "rank": 32,
                "alpha": 32,
                "learning_rate": 1e-4,
                "steps": 3000,
                "repeats": 1,
                "batch_size": 1,
                "frames": 1,
                "dataset_mode": "clip_caption",
                "dataset_resolutions": [512, 768, 1024],
                "control_type": "none",
                "attention_strength": 1.0,
                "target_fps": 24,
                "caption_dropout": 0.05,
                "weight_decay": 0.0001,
                "timestep_type": "weighted",
                "timestep_bias": "balanced",
                "loss_type": "mse",
                "sample_every": 250,
                "sample_sampler": "flowmatch",
                "sample_guidance_scale": 4,
                "sample_steps": 25,
                "sample_size": "768x768",
                "cache_text_embeddings": False,
            },
            "motion": {
                "mode_label": "Motion LoRA",
                "training_mode": "lora",
                "rank": 32,
                "alpha": 32,
                "learning_rate": 1e-4,
                "steps": 3000,
                "repeats": 1,
                "batch_size": 1,
                "frames": 49,
                "dataset_mode": "clip_caption",
                "dataset_resolutions": [512, 768, 1024],
                "control_type": "none",
                "attention_strength": 1.0,
                "target_fps": 24,
                "caption_dropout": 0.05,
                "weight_decay": 0.0001,
                "timestep_type": "weighted",
                "timestep_bias": "balanced",
                "loss_type": "mse",
                "sample_every": 250,
                "sample_sampler": "flowmatch",
                "sample_guidance_scale": 4,
                "sample_steps": 25,
                "sample_size": "768x768",
                "cache_text_embeddings": False,
            },
            "audio_video": {
                "mode_label": "Audio-Video LoRA",
                "training_mode": "audio_video_lora",
                "rank": 16,
                "alpha": 8,
                "learning_rate": 1.5e-5,
                "steps": 2400,
                "repeats": 3,
                "batch_size": 1,
                "dataset_mode": "audio_video_pairs",
                "control_type": "none",
                "attention_strength": 1.0,
                "target_fps": 24,
            },
            "ic_lora": {
                "mode_label": "IC-LoRA",
                "training_mode": "ic_lora",
                "rank": 32,
                "alpha": 32,
                "learning_rate": 1e-4,
                "steps": 3000,
                "repeats": 1,
                "batch_size": 1,
                "frames": 49,
                "dataset_mode": "paired_control_video",
                "dataset_resolutions": [512, 768, 1024],
                "control_type": "union",
                "attention_strength": 1.0,
                "target_fps": 24,
                "caption_dropout": 0.05,
                "weight_decay": 0.0001,
                "timestep_type": "weighted",
                "timestep_bias": "balanced",
                "loss_type": "mse",
                "sample_every": 250,
                "sample_sampler": "flowmatch",
                "sample_guidance_scale": 4,
                "sample_steps": 25,
                "sample_size": "768x768",
                "cache_text_embeddings": False,
            },
        },
    },
    "Anima": {
        "label": "ANIMA",
        "base": "Anima / anime SD family",
        "trainer": "kohya_ss",
        "trainer_label": "kohya-ss sd-scripts",
        "script": "train_network.py",
        "resolution": 768,
        "caption": "Use tag-style captions for anime and keep character tags stable.",
        "templates": {
            "character": {"rank": 16, "alpha": 16, "learning_rate": 1e-4, "steps": 1800, "repeats": 12, "batch_size": 2},
            "style": {"rank": 8, "alpha": 8, "learning_rate": 8e-5, "steps": 1200, "repeats": 8, "batch_size": 2},
        },
    },
}


DEVICE_PROFILES = {
    "auto": {"label": "Auto", "description": "Uses detected VRAM and Nexus runtime policy."},
    "low": {
        "label": "Low VRAM 6-8GB",
        "memory_policy": "shared",
        "precision": "fp16",
        "batch_size": 1,
        "gradient_accumulation": 4,
        "max_rank": 8,
        "optimizer": "adamw8bit",
        "cache_latents": True,
        "gradient_checkpointing": True,
    },
    "balanced": {
        "label": "Balanced 10-16GB",
        "memory_policy": "shared",
        "precision": "bf16",
        "batch_size": 1,
        "gradient_accumulation": 2,
        "max_rank": 16,
        "optimizer": "adamw8bit",
        "cache_latents": True,
        "gradient_checkpointing": True,
    },
    "high": {
        "label": "High VRAM 20-24GB",
        "memory_policy": "gpu_only",
        "precision": "bf16",
        "batch_size": 2,
        "gradient_accumulation": 1,
        "max_rank": 32,
        "optimizer": "adamw",
        "cache_latents": True,
        "gradient_checkpointing": False,
    },
    "video": {
        "label": "Video 32GB+",
        "memory_policy": "gpu_only",
        "precision": "bf16",
        "batch_size": 1,
        "gradient_accumulation": 1,
        "max_rank": 32,
        "optimizer": "adamw8bit",
        "cache_latents": True,
        "gradient_checkpointing": True,
    },
}


def _safe_name(value: Any, fallback: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value or "").strip()).strip("._-")
    return text[:80] or fallback


def _number(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(maximum, parsed))


def _int(value: Any, default: int, minimum: int, maximum: int) -> int:
    return int(round(_number(value, default, minimum, maximum)))


def _perceptual_job_config(preset: dict[str, Any], plan: dict[str, Any]) -> dict[str, Any]:
    support = dict(preset.get("perceptual") or {})
    options = dict(plan.get("perceptual") or {})
    enabled = bool(options.get("enabled")) and bool(support.get("compatible"))
    anchor = str(options.get("anchor") or PERCEPTUAL_DEFAULTS["anchor"]).lower()
    if anchor not in {"none", "depth"}:
        anchor = "depth"
    image_mode = str(options.get("image_mode") or PERCEPTUAL_DEFAULTS["image_mode"]).lower()
    if image_mode not in {"auto", "single", "multi"}:
        image_mode = "auto"
    default_sigma = (
        PERCEPTUAL_DEFAULTS["single_image_sigma"]
        if image_mode == "single"
        else PERCEPTUAL_DEFAULTS["multi_image_sigma"]
    )
    sigma = _number(options.get("weight_noise_sigma"), float(default_sigma), 0.0, 0.1)
    depth_loss_weight = _number(
        options.get("depth_loss_weight"),
        0.0 if anchor == "none" else float(PERCEPTUAL_DEFAULTS["depth_loss_weight"]),
        0.0,
        1.0,
    )
    weight_noise_enabled = bool(options.get("weight_noise_enabled", PERCEPTUAL_DEFAULTS["weight_noise_enabled"]))
    if not enabled:
        depth_loss_weight = 0.0
        weight_noise_enabled = False
    return {
        "enabled": enabled,
        "compatible": bool(support.get("compatible")),
        "support_level": str(support.get("support_level") or "unsupported"),
        "recommended_for": support.get("recommended_for") or [],
        "notes": str(support.get("notes") or "Not advertised as compatible by ai-toolkit-perceptual."),
        "source_url": "https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual",
        "anchor": anchor,
        "image_mode": image_mode,
        "depth_consistency": {
            "loss_weight": depth_loss_weight,
            "model_id": str(options.get("depth_model_id") or PERCEPTUAL_DEFAULTS["depth_model_id"]),
            "mask_source": str(options.get("depth_mask_source") or PERCEPTUAL_DEFAULTS["depth_mask_source"]),
            "loss_min_t": _number(options.get("loss_min_t"), float(PERCEPTUAL_DEFAULTS["loss_min_t"]), 0.0, 1.0),
            "loss_max_t": _number(options.get("loss_max_t"), float(PERCEPTUAL_DEFAULTS["loss_max_t"]), 0.0, 1.0),
            "preview_every": _int(options.get("preview_every"), int(PERCEPTUAL_DEFAULTS["preview_every"]), 1, 10000),
        },
        "loss_split": "diffusion_depth" if enabled and anchor == "depth" and depth_loss_weight > 0 else None,
        "weight_noise": {
            "enabled": weight_noise_enabled,
            "mode": str(options.get("weight_noise_mode") or PERCEPTUAL_DEFAULTS["weight_noise_mode"]),
            "sigma": sigma,
            "log_every": _int(options.get("weight_noise_log_every"), int(PERCEPTUAL_DEFAULTS["weight_noise_log_every"]), 1, 10000),
        },
    }


def train_lora_root(settings: NexusSettings) -> Path:
    root = settings.project_root / "training" / "lora"
    root.mkdir(parents=True, exist_ok=True)
    return root


def train_lora_job_root(settings: NexusSettings, job_id: str) -> Path:
    job_root = train_lora_root(settings) / "jobs" / _safe_name(job_id, "job")
    job_root.mkdir(parents=True, exist_ok=True)
    return job_root


def _detected_vram_gb(settings: NexusSettings, memory_snapshot: dict[str, Any] | None) -> float:
    configured = settings.runtime.gpu_memory_gb
    if configured:
        return float(configured)
    torch_info = ((memory_snapshot or {}).get("torch") or {})
    total = torch_info.get("cuda_total_vram_bytes") or 0
    try:
        return float(total) / (1024**3)
    except (TypeError, ValueError):
        return 0.0


def recommended_device_profile(settings: NexusSettings, memory_snapshot: dict[str, Any] | None = None) -> str:
    vram = _detected_vram_gb(settings, memory_snapshot)
    if vram >= 32:
        return "video"
    if vram >= 20:
        return "high"
    if vram >= 10:
        return "balanced"
    return "low"


def build_train_lora_catalog(settings: NexusSettings, memory_snapshot: dict[str, Any] | None = None) -> dict[str, Any]:
    recommended = recommended_device_profile(settings, memory_snapshot)
    template_order = ["SD", "XL", "Illustrious", "Pony", "Flux", "Flux2", "Flux2Klein", "Qwen", "Wan", "LTX", "Anima", "ZImageTurbo", "Lumina"]
    return {
        "templates": TRAIN_LORA_PRESETS,
        "template_order": [key for key in template_order if key in TRAIN_LORA_PRESETS],
        "device_profiles": DEVICE_PROFILES,
        "perceptual_defaults": PERCEPTUAL_DEFAULTS,
        "memory_policies": {
            "auto": "Auto from selected device profile",
            "shared": "Shared VRAM: CPU/RAM offload, safer for low VRAM, slower.",
            "gpu_only": "GPU only: faster when the model fits in VRAM.",
        },
        "recommended_device": recommended,
        "runtime_vram_policy": settings.runtime.vram_policy,
        "detected_vram_gb": round(_detected_vram_gb(settings, memory_snapshot), 2),
        "paths": {
            "root": str(train_lora_root(settings)),
            "jobs": str(train_lora_root(settings) / "jobs"),
            "outputs": str(train_lora_root(settings) / "outputs"),
        },
        "sources": TRAIN_LORA_SOURCES,
    }


def _trainer_candidates(settings: NexusSettings, trainer: str, script: str | None) -> list[tuple[Path, list[str]]]:
    python_bin = str(runtime_python(settings))
    root = settings.project_root
    runtime = settings.runtime_dir
    candidates: list[tuple[Path, list[str]]] = []
    if trainer == "kohya_ss":
        script_name = script or "train_network.py"
        for base in (runtime / "sd-scripts", runtime / "kohya_ss" / "sd-scripts", root / "sd-scripts"):
            candidates.append((base, [python_bin, str(base / script_name)]))
    elif trainer == "ai_toolkit":
        for base in (runtime / "ai-toolkit", runtime / "ai_toolkit", root / "ai-toolkit"):
            candidates.append((base, [python_bin, str(base / "run.py")]))
    elif trainer == "ai_toolkit_perceptual":
        for base in (
            runtime / "ai-toolkit-perceptual",
            runtime / "ai_toolkit_perceptual",
            root / "ai-toolkit-perceptual",
        ):
            candidates.append((base, [python_bin, str(base / "run.py")]))
    elif trainer == "ltx_trainer":
        uv = shutil.which("uv")
        if not uv:
            return candidates
        for base in (runtime / "LTX-2", root / "LTX-2"):
            candidates.append((base, [uv, "run", "python", "packages/ltx-trainer/scripts/train.py"]))
    elif trainer == "musubi_tuner":
        for base in (runtime / "musubi-tuner", runtime / "musubi_tuner", root / "musubi-tuner"):
            candidates.append((base, [python_bin, str(base / "wan_train_network.py")]))
    elif trainer == "simpletuner":
        for base in (runtime / "SimpleTuner", runtime / "simpletuner", root / "SimpleTuner"):
            candidates.append((base, [python_bin, str(base / "train.py")]))
    return candidates


def _trainer_config_args(trainer: str, config_path: Path) -> list[str]:
    if trainer == "ltx_trainer":
        return [str(config_path)]
    return ["--config_file", str(config_path)]


def _parse_ltx_size(value: Any, default: tuple[int, int] = (768, 768)) -> tuple[int, int]:
    text = str(value or "").strip().lower()
    match = re.search(r"(\d+)\s*x\s*(\d+)", text)
    if match:
        width = int(match.group(1))
        height = int(match.group(2))
    else:
        number_match = re.search(r"\d+", text)
        if not number_match:
            return default
        width = height = int(number_match.group(0))
    width = max(32, (width // 32) * 32)
    height = max(32, (height // 32) * 32)
    return width, height


def _ltx_frame_count(value: Any) -> int:
    frames = _int(value, 1, 1, 257)
    if frames == 1:
        return 1
    return max(9, ((frames - 1) // 8) * 8 + 1)


def _ltx_target_modules(training_mode: str, control_type: str) -> list[str]:
    if training_mode == "audio_video_lora":
        return ["to_k", "to_q", "to_v", "to_out.0"]
    if training_mode == "ic_lora" or control_type != "none":
        return [
            "attn1.to_k",
            "attn1.to_q",
            "attn1.to_v",
            "attn1.to_out.0",
            "attn2.to_k",
            "attn2.to_q",
            "attn2.to_v",
            "attn2.to_out.0",
            "ff.net.0.proj",
            "ff.net.2",
        ]
    return [
        "attn1.to_k",
        "attn1.to_q",
        "attn1.to_v",
        "attn1.to_out.0",
        "attn2.to_k",
        "attn2.to_q",
        "attn2.to_v",
        "attn2.to_out.0",
    ]


def _ltx_trainer_config(config: dict[str, Any], dataset_dir: Path, output_dir: Path) -> dict[str, Any]:
    ltx = dict(config.get("ltx") or {})
    training_mode = str(ltx.get("training_mode") or "lora").lower()
    control_type = str(ltx.get("control_type") or "none").lower()
    text_encoder_path = str(ltx.get("text_encoder_path") or "").strip()
    sample_width, sample_height = _parse_ltx_size(ltx.get("sample_size") or config.get("resolution"))
    frames = _ltx_frame_count(config.get("frames"))
    precision = str(config.get("precision") or "bf16").lower()
    if precision not in {"no", "fp16", "bf16"}:
        precision = "bf16"
    optimizer = str(config.get("optimizer") or "adamw").lower()
    if optimizer not in {"adamw", "adamw8bit"}:
        optimizer = "adamw8bit" if "8" in optimizer else "adamw"
    strategy: dict[str, Any]
    if training_mode == "ic_lora" or control_type != "none":
        strategy = {
            "name": "video_to_video",
            "first_frame_conditioning_p": 0.2,
            "reference_latents_dir": "reference_latents",
        }
    else:
        strategy = {
            "name": "text_to_video",
            "first_frame_conditioning_p": 0.5 if frames > 1 else 0.1,
            "with_audio": training_mode == "audio_video_lora" or str(ltx.get("dataset_mode") or "") == "audio_video_pairs",
            "audio_latents_dir": "audio_latents",
        }
    model_config: dict[str, Any] = {
        "model_path": str(config.get("base_model_path") or ""),
        "training_mode": "lora",
        "load_checkpoint": str(config.get("resume_from") or "") or None,
    }
    if text_encoder_path:
        model_config["text_encoder_path"] = text_encoder_path
    return {
        "model": model_config,
        "lora": {
            "rank": int(config["rank"]),
            "alpha": int(config["alpha"]),
            "dropout": 0.0,
            "target_modules": _ltx_target_modules(training_mode, control_type),
        },
        "training_strategy": strategy,
        "optimization": {
            "learning_rate": float(config["learning_rate"]),
            "steps": int(config["steps"]),
            "batch_size": int(config["batch_size"]),
            "gradient_accumulation_steps": int(config["gradient_accumulation"]),
            "max_grad_norm": 1.0,
            "optimizer_type": optimizer,
            "scheduler_type": "linear",
            "scheduler_params": {},
            "enable_gradient_checkpointing": bool(config.get("gradient_checkpointing", True)),
        },
        "acceleration": {
            "mixed_precision_mode": precision,
            "quantization": None,
            "load_text_encoder_in_8bit": bool(config.get("low_vram", False)),
        },
        "data": {
            "preprocessed_data_root": str(dataset_dir),
            "num_dataloader_workers": 0 if bool(config.get("low_vram", False)) else 2,
        },
        "validation": {
            "prompts": [str(config.get("caption") or config.get("trigger_word") or "A video of the training subject.")],
            "negative_prompt": "worst quality, inconsistent motion, blurry, jittery, distorted",
            "images": None,
            "reference_videos": None,
            "video_dims": [sample_width, sample_height, frames],
            "frame_rate": float(ltx.get("target_fps") or 24),
            "seed": 42,
            "inference_steps": int(ltx.get("sample_steps") or 25),
            "interval": None
            if strategy["name"] == "video_to_video"
            else int(ltx.get("sample_every") or config.get("save_every_n_steps") or 250),
            "videos_per_prompt": 1,
            "guidance_scale": float(ltx.get("sample_guidance_scale") or 3.0),
            "stg_scale": 1.0,
            "stg_blocks": [29],
            "stg_mode": "stg_av" if training_mode == "audio_video_lora" else "stg_v",
            "generate_audio": training_mode == "audio_video_lora",
            "skip_initial_validation": True,
            "include_reference_in_output": False,
        },
        "checkpoints": {
            "interval": int(config["save_every_n_steps"]),
            "keep_last_n": -1,
        },
        "hub": {
            "push_to_hub": False,
            "hub_model_id": None,
        },
        "flow_matching": {
            "timestep_sampling_mode": "shifted_logit_normal",
            "timestep_sampling_params": {},
        },
        "wandb": {
            "enabled": False,
            "project": "ltx-2-trainer",
            "entity": None,
            "tags": ["nexus-bta", "ltx2", "lora"],
            "log_validation_videos": False,
        },
        "seed": 42,
        "output_dir": str(output_dir),
    }


def _ltx_launch_issue(config: dict[str, Any]) -> str:
    model_path = str(config.get("base_model_path") or "").strip()
    if not model_path:
        return "Select a local LTX-2.3 model checkpoint (.safetensors) before launching the LTX-2 trainer."
    model_file = Path(model_path)
    if not model_file.exists():
        return f"LTX-2 trainer model checkpoint does not exist: {model_path}"
    model_name = model_file.name.lower()
    if any(token in model_name for token in ("vae", "upscaler", "lora", "text_encoder", "gemma")):
        return (
            "The selected LTX base model looks like an auxiliary file, not the main model checkpoint: "
            f"{model_file.name}. Select the LTX-2.3 22B model checkpoint, not a VAE, LoRA, upscaler, or text encoder."
        )
    text_encoder_path = str((config.get("ltx") or {}).get("text_encoder_path") or "").strip()
    if not text_encoder_path:
        return "Set the Gemma Text Encoder folder for LTX-2.3 training before launching the LTX-2 trainer."
    if not Path(text_encoder_path).exists():
        return f"LTX-2.3 Gemma Text Encoder path does not exist: {text_encoder_path}"
    return ""


def resolve_trainer_runner(settings: NexusSettings, preset: dict[str, Any], config_path: Path) -> dict[str, Any]:
    trainer = str(preset.get("trainer") or "")
    script = str(preset.get("script") or "") or None
    for cwd, command in _trainer_candidates(settings, trainer, script):
        executable = Path(command[1]) if len(command) > 1 and command[1].endswith(".py") else cwd
        if cwd.exists() and (not executable.suffix or executable.exists()):
            final_command = [*command, *_trainer_config_args(trainer, config_path)]
            return {"available": True, "cwd": str(cwd), "command": final_command}
    return {
        "available": False,
        "cwd": "",
        "command": [],
        "install_hint": _install_hint(trainer),
    }


_TRAINER_REPOSITORIES: dict[str, tuple[str, str, str]] = {
    "kohya_ss": ("https://github.com/kohya-ss/sd-scripts.git", "sd-scripts", "train_network.py"),
    "ai_toolkit": ("https://github.com/ostris/ai-toolkit.git", "ai-toolkit", "run.py"),
    "ai_toolkit_perceptual": (
        "https://github.com/BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual.git",
        "ai-toolkit-perceptual",
        "run.py",
    ),
    "ltx_trainer": ("https://github.com/Lightricks/LTX-2.git", "LTX-2", "packages/ltx-trainer/scripts/train.py"),
    "musubi_tuner": ("https://github.com/kohya-ss/musubi-tuner.git", "musubi-tuner", "wan_train_network.py"),
    "simpletuner": ("https://github.com/bghira/SimpleTuner.git", "SimpleTuner", "train.py"),
}


def trainer_dependency_status(settings: NexusSettings, trainer: str, script: str | None = None) -> dict[str, Any]:
    """Return the local install target used by the selected LoRA runner."""
    normalized = str(trainer or "").strip().lower()
    repo = _TRAINER_REPOSITORIES.get(normalized)
    if not repo:
        return {
            "trainer": normalized,
            "available": False,
            "install_dir": "",
            "python": str(runtime_python(settings)),
            "repo_url": "",
            "install_hint": _install_hint(normalized),
        }
    repo_url, folder, default_script = repo
    requested_script = str(script or default_script).replace("\\", "/")
    requested_path = Path(requested_script)
    if requested_path.is_absolute() or ".." in requested_path.parts:
        raise ValueError("Trainer script path must stay inside the selected trainer folder.")
    install_dir = settings.runtime_dir / folder
    script_path = install_dir / requested_script
    requirements_path = install_dir / "requirements.txt"
    return {
        "trainer": normalized,
        "available": install_dir.is_dir() and script_path.is_file(),
        "install_dir": str(install_dir),
        "python": str(runtime_python(settings)),
        "repo_url": repo_url,
        "script": requested_script,
        "requirements": str(requirements_path),
        "requirements_available": requirements_path.is_file(),
        "install_hint": _install_hint(normalized),
    }


def install_trainer_dependency(settings: NexusSettings, trainer: str, script: str | None = None) -> dict[str, Any]:
    """Clone/update a trainer inside the configured runtime and install its requirements."""
    status = trainer_dependency_status(settings, trainer, script)
    normalized = status["trainer"]
    repo = _TRAINER_REPOSITORIES.get(normalized)
    if not repo:
        raise ValueError(f"Unsupported LoRA trainer: {trainer}")
    git = shutil.which("git")
    if not git:
        raise RuntimeError("Git is required to download the selected LoRA trainer.")
    python_exe = Path(status["python"])
    if not python_exe.is_file():
        raise RuntimeError(f"Nexus runtime Python was not found: {python_exe}")

    settings.runtime_dir.mkdir(parents=True, exist_ok=True)
    target = Path(status["install_dir"])
    repo_url = str(repo[0])
    if target.exists():
        if not (target / ".git").is_dir():
            raise RuntimeError(f"Install target exists but is not a Git checkout: {target}")
        dirty = subprocess.run(
            [git, "-C", str(target), "status", "--porcelain"],
            cwd=str(settings.project_root),
            capture_output=True,
            text=True,
            timeout=60,
        )
        if dirty.returncode != 0:
            raise RuntimeError((dirty.stderr or dirty.stdout or "Unable to inspect trainer checkout.").strip())
        if not dirty.stdout.strip():
            pulled = subprocess.run(
                [git, "-C", str(target), "pull", "--ff-only"],
                cwd=str(settings.project_root),
                capture_output=True,
                text=True,
                timeout=900,
            )
            if pulled.returncode != 0:
                raise RuntimeError((pulled.stderr or pulled.stdout or "Trainer update failed.").strip()[-2000:])
    else:
        cloned = subprocess.run(
            [git, "clone", "--depth", "1", repo_url, str(target)],
            cwd=str(settings.runtime_dir),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if cloned.returncode != 0:
            raise RuntimeError((cloned.stderr or cloned.stdout or "Trainer download failed.").strip()[-2000:])

    requirements = target / "requirements.txt"
    if requirements.is_file():
        installed = subprocess.run(
            [str(python_exe), "-m", "pip", "install", "-r", str(requirements)],
            cwd=str(target),
            capture_output=True,
            text=True,
            timeout=1800,
        )
        if installed.returncode != 0:
            raise RuntimeError((installed.stderr or installed.stdout or "Trainer requirements installation failed.").strip()[-3000:])

    refreshed = trainer_dependency_status(settings, normalized, script)
    refreshed["message"] = f"{normalized} installed in {target}."
    return refreshed


def _install_hint(trainer: str) -> str:
    hints = {
        "kohya_ss": "Clone kohya-ss/sd-scripts into runtime/sd-scripts and install its requirements.",
        "ai_toolkit": "Clone ostris/ai-toolkit into runtime/ai-toolkit and install requirements.",
        "ai_toolkit_perceptual": "Clone BuffaloBuffaloBuffaloBuffalo/ai-toolkit-perceptual into runtime/ai-toolkit-perceptual and install requirements.",
        "ltx_trainer": "Clone Lightricks/LTX-2 into runtime/LTX-2 and run uv sync.",
        "musubi_tuner": "Install musubi-tuner into runtime/musubi-tuner for Wan video LoRA jobs.",
        "simpletuner": "Clone SimpleTuner into runtime/SimpleTuner and install requirements.",
    }
    return hints.get(trainer, "Install the selected trainer before launching.")


def _toml_string(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(str(value))


def write_kohya_dataset_toml(path: Path, dataset_dir: Path, config: dict[str, Any]) -> None:
    lines = [
        "[general]",
        "shuffle_caption = true",
        "caption_extension = \".txt\"",
        "keep_tokens = 1",
        "",
        "[[datasets]]",
        f"resolution = {_toml_string(config['resolution'])}",
        "batch_size = 1",
        "enable_bucket = true",
        "bucket_no_upscale = false",
        "",
        "[[datasets.subsets]]",
        f"image_dir = {_toml_string(str(dataset_dir))}",
        f"num_repeats = {int(config['repeats'])}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_train_lora_job(
    settings: NexusSettings,
    job_id: str,
    plan: dict[str, Any],
    saved_files: list[dict[str, Any]],
    memory_snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    preset_key = str(plan.get("preset") or "SD")
    preset = TRAIN_LORA_PRESETS.get(preset_key) or TRAIN_LORA_PRESETS["SD"]
    mode = str(plan.get("mode") or plan.get("type") or "character").lower()
    preset_templates = preset.get("templates") or {}
    valid_modes = set(preset_templates) or {"character", "style"}
    if mode not in valid_modes:
        mode = "character" if "character" in valid_modes else sorted(valid_modes)[0]
    base_defaults = dict(preset_templates.get(mode) or {})
    requested_device = str(plan.get("device_profile") or "auto")
    if requested_device == "auto" or requested_device not in DEVICE_PROFILES:
        requested_device = recommended_device_profile(settings, memory_snapshot)
    device = dict(DEVICE_PROFILES.get(requested_device) or DEVICE_PROFILES["low"])
    training = dict(plan.get("training") or {})
    is_ltx_trainer = str(preset.get("trainer") or "") == "ltx_trainer"
    base_rank = int(base_defaults.get("rank", 16))
    rank_cap = int(device.get("max_rank", 16))
    if is_ltx_trainer:
        rank_cap = max(rank_cap, 64, base_rank)
    rank_default = base_rank if is_ltx_trainer else min(base_rank, rank_cap)
    memory_policy = str(training.get("memory_policy") or "").strip().lower()
    if memory_policy not in {"shared", "gpu_only"}:
        memory_policy = str(device.get("memory_policy") or settings.runtime.vram_policy or "shared").strip().lower()
    if memory_policy not in {"shared", "gpu_only"}:
        memory_policy = "shared"
    perceptual = _perceptual_job_config(preset, plan)
    trainer_name = "ai_toolkit_perceptual" if perceptual["enabled"] else str(preset["trainer"])
    trainer_label = "AI Toolkit Perceptual" if perceptual["enabled"] else str(preset["trainer_label"])
    ltx_options = dict(plan.get("ltx") or {}) if is_ltx_trainer else {}
    ltx_training_mode = str(ltx_options.get("training_mode") or base_defaults.get("training_mode") or "lora")
    ltx_control_type = str(ltx_options.get("control_type") or base_defaults.get("control_type") or "none").lower()
    ltx_dataset_mode = str(ltx_options.get("dataset_mode") or base_defaults.get("dataset_mode") or "clip_caption")
    ltx_attention_strength = _number(
        ltx_options.get("attention_strength"),
        float(base_defaults.get("attention_strength", 1.0)),
        0.0,
        1.0,
    )
    ltx_target_fps = _int(ltx_options.get("target_fps"), int(base_defaults.get("target_fps", 24)), 1, 60)
    config = {
        "preset": preset_key,
        "preset_label": preset["label"],
        "mode": mode,
        "mode_label": base_defaults.get("mode_label", mode.replace("_", " ").title()),
        "base_model": preset["base"],
        "trainer": trainer_name,
        "trainer_label": trainer_label,
        "base_trainer": preset["trainer"],
        "base_trainer_label": preset["trainer_label"],
        "base_model_path": str(plan.get("base_model_path") or ""),
        "resume_from": str(plan.get("resume_from") or ""),
        "trigger_word": _safe_name(plan.get("trigger_word"), f"nexus_{preset_key.lower()}_{mode}"),
        "output_name": _safe_name(plan.get("output_name"), f"nexus_{preset_key.lower()}_{mode}_lora"),
        "resolution": training.get("resolution") or base_defaults.get("resolution") or preset.get("resolution") or 1024,
        "frames": _int(training.get("frames"), int(base_defaults.get("frames") or preset.get("frames") or 1), 1, 257),
        "rank": _int(training.get("rank"), rank_default, 1, rank_cap),
        "alpha": _int(training.get("alpha"), int(base_defaults.get("alpha", rank_default)), 1, 128),
        "learning_rate": _number(training.get("learning_rate"), float(base_defaults.get("learning_rate", 1e-4)), 1e-7, 1e-3),
        "steps": _int(training.get("steps"), int(base_defaults.get("steps", 1200)), 100, 200000),
        "save_every_n_steps": _int(training.get("save_every_n_steps"), int(base_defaults.get("save_every_n_steps", 250)), 25, 50000),
        "repeats": _int(training.get("repeats"), int(base_defaults.get("repeats", 8)), 1, 200),
        "batch_size": _int(training.get("batch_size"), int(base_defaults.get("batch_size", device.get("batch_size", 1))), 1, 16),
        "gradient_accumulation": _int(training.get("gradient_accumulation"), int(device.get("gradient_accumulation", 1)), 1, 64),
        "precision": str(training.get("precision") or device.get("precision") or "fp16"),
        "optimizer": str(training.get("optimizer") or device.get("optimizer") or "adamw8bit"),
        "memory_policy": memory_policy,
        "low_vram": memory_policy == "shared",
        "cache_latents": bool(training.get("cache_latents", device.get("cache_latents", True))),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", device.get("gradient_checkpointing", True))),
        "caption": str(plan.get("caption") or preset.get("caption") or ""),
        "source_dir": str(plan.get("source_dir") or ""),
        "device_profile": requested_device,
    }
    config["perceptual"] = perceptual
    if is_ltx_trainer:
        config["ltx"] = {
            "training_mode": ltx_training_mode,
            "control_type": ltx_control_type,
            "dataset_mode": ltx_dataset_mode,
            "dataset_resolutions": base_defaults.get("dataset_resolutions") or [],
            "caption_dropout": float(base_defaults.get("caption_dropout", 0.05)),
            "weight_decay": float(base_defaults.get("weight_decay", 0.0001)),
            "timestep_type": str(base_defaults.get("timestep_type", "weighted")),
            "timestep_bias": str(base_defaults.get("timestep_bias", "balanced")),
            "loss_type": str(base_defaults.get("loss_type", "mse")),
            "sample_every": int(base_defaults.get("sample_every", config["save_every_n_steps"])),
            "sample_sampler": str(base_defaults.get("sample_sampler", "flowmatch")),
            "sample_guidance_scale": float(base_defaults.get("sample_guidance_scale", 4)),
            "sample_steps": int(base_defaults.get("sample_steps", 25)),
            "sample_size": str(base_defaults.get("sample_size", "768x768")),
            "cache_text_embeddings": bool(base_defaults.get("cache_text_embeddings", False)),
            "attention_strength": ltx_attention_strength,
            "target_fps": ltx_target_fps,
            "text_encoder_path": str(ltx_options.get("text_encoder_path") or ""),
            "comfy_nodes": [
                "LTXICLoRALoaderModelOnly",
                "LTXAddVideoICLoRAGuide",
                "LTXVPreprocess",
            ] if ltx_training_mode == "ic_lora" or ltx_control_type != "none" else [],
            "notes": (
                "IC-LoRA expects paired target/control media with matched resolution and FPS."
                if ltx_training_mode == "ic_lora" or ltx_control_type != "none"
                else "Standard LTX LoRA uses clips or frames with captions."
            ),
        }

    job_root = train_lora_job_root(settings, job_id)
    dataset_dir = job_root / "dataset"
    output_dir = train_lora_root(settings) / "outputs" / config["output_name"]
    custom_output_dir = str(plan.get("output_dir") or "").strip()
    if custom_output_dir and not custom_output_dir.startswith("[browser-folder]"):
        output_dir = Path(custom_output_dir).expanduser() / config["output_name"]
    output_dir.mkdir(parents=True, exist_ok=True)
    config["dataset_dir"] = str(dataset_dir)
    config["output_dir"] = str(output_dir)
    config["uploaded_files"] = saved_files
    config_path = job_root / "train_lora_config.json"
    trainer_config = _ltx_trainer_config(config, dataset_dir, output_dir) if is_ltx_trainer else config
    config_path.write_text(json.dumps(trainer_config, indent=2), encoding="utf-8")

    dataset_toml = None
    if config["trainer"] == "kohya_ss":
        dataset_toml = job_root / "dataset.toml"
        write_kohya_dataset_toml(dataset_toml, dataset_dir, config)
    readme_path = job_root / "README.md"
    readme_path.write_text(_job_readme(config, config_path, dataset_toml), encoding="utf-8")
    runner_preset = {**preset, "trainer": config["trainer"], "trainer_label": config["trainer_label"], "script": ""}
    runner = resolve_trainer_runner(settings, runner_preset, config_path)
    ltx_launch_issue = _ltx_launch_issue(config) if is_ltx_trainer else ""
    if ltx_launch_issue:
        runner = {**runner, "available": False, "install_hint": ltx_launch_issue}
    terminal_log_path = job_root / "train_lora_terminal.log"
    terminal_log_path.write_text(_job_terminal_log(config, runner, saved_files), encoding="utf-8")
    status = "prepared"
    message = "Train LoRA job prepared. Enable launch after installing the selected trainer."
    if plan.get("launch") and not runner.get("available"):
        status = "blocked"
        message = runner.get("install_hint") or message
    elif plan.get("launch") and runner.get("available"):
        status = "queued"
        message = "Train LoRA job queued."
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "job_id": job_id,
        "status": status,
        "progress": 20 if status == "prepared" else 0,
        "message": message,
        "error": None,
        "preset": preset_key,
        "mode": mode,
        "config": config,
        "paths": {
            "job": str(job_root),
            "dataset": str(dataset_dir),
            "config": str(config_path),
            "dataset_toml": str(dataset_toml) if dataset_toml else "",
            "readme": str(readme_path),
            "output": str(output_dir),
            "terminal_log": str(terminal_log_path),
        },
        "runner": runner,
        "created_at": now,
        "updated_at": now,
    }


def _job_readme(config: dict[str, Any], config_path: Path, dataset_toml: Path | None) -> str:
    lines = [
        f"# {config['output_name']}",
        "",
        f"Preset: {config['preset_label']} / {config.get('mode_label') or config['mode']}",
        f"Trainer: {config['trainer_label']}",
        f"Trigger: `{config['trigger_word']}`",
        f"Config: `{config_path}`",
    ]
    if dataset_toml:
        lines.append(f"Dataset TOML: `{dataset_toml}`")
    lines.extend(
        [
            "",
            "LTX/Comfy route:" if config.get("ltx") else "",
            *((
                [
                    f"- Training mode: `{config['ltx']['training_mode']}`",
                    f"- Control type: `{config['ltx']['control_type']}`",
                    f"- Dataset mode: `{config['ltx']['dataset_mode']}`",
                    f"- Dataset resolutions: `{', '.join(str(item) for item in config['ltx'].get('dataset_resolutions') or ['configured resolution'])}`",
                    f"- Caption dropout: `{config['ltx']['caption_dropout']}`",
                    f"- Timesteps/loss: `{config['ltx']['timestep_type']} / {config['ltx']['timestep_bias']} / {config['ltx']['loss_type']}`",
                    f"- Validation: every `{config['ltx']['sample_every']}` steps, `{config['ltx']['sample_sampler']}`, guidance `{config['ltx']['sample_guidance_scale']}`, `{config['ltx']['sample_steps']}` steps, `{config['ltx']['sample_size']}`",
                    f"- Cache text embeddings: `{config['ltx']['cache_text_embeddings']}`",
                    f"- Attention strength: `{config['ltx']['attention_strength']}`",
                    f"- Target FPS: `{config['ltx']['target_fps']}`",
                    f"- Comfy nodes: `{', '.join(config['ltx'].get('comfy_nodes') or ['standard LTX LoRA route'])}`",
                ]
            ) if config.get("ltx") else []),
            "",
            "Install notes:",
            "- SD/SDXL/Anima: kohya-ss sd-scripts.",
            "- FLUX/Qwen: AI Toolkit.",
            "- LTX: LTX-2 trainer with cached latents/embeddings; IC-LoRA pairs target video with control signals.",
            "- Wan: musubi-tuner compatible Wan LoRA trainer.",
            "- Perceptual: ai-toolkit-perceptual fork for compatible SDXL / Flux 2 Klein / Z-Image Turbo / LTX jobs.",
        ]
    )
    if (config.get("perceptual") or {}).get("enabled"):
        perceptual = config["perceptual"]
        lines.extend(
            [
                "",
                "Perceptual route:",
                f"- Anchor: `{perceptual['anchor']}`",
                f"- Image mode: `{perceptual['image_mode']}`",
                f"- Depth weight/model: `{perceptual['depth_consistency']['loss_weight']}` / `{perceptual['depth_consistency']['model_id']}`",
                f"- Depth mask: `{perceptual['depth_consistency']['mask_source']}`",
                f"- Loss split: `{perceptual['loss_split'] or 'auto/off'}`",
                f"- Weight noise: `{perceptual['weight_noise']['enabled']}` sigma `{perceptual['weight_noise']['sigma']}`",
                f"- Support level: `{perceptual['support_level']}`",
            ]
        )
    return "\n".join(line for line in lines if line != "") + "\n"


def _job_terminal_log(config: dict[str, Any], runner: dict[str, Any], saved_files: list[dict[str, Any]]) -> str:
    lines = [
        f"[{datetime.now().isoformat(timespec='seconds')}] Nexus Train LoRA prepared",
        f"preset={config['preset_label']} mode={config['mode']} trainer={config['trainer_label']}",
        f"trigger={config['trigger_word']} output={config['output_name']}",
        f"base_model_path={config['base_model_path'] or '-'} resume_from={config['resume_from'] or '-'}",
        f"steps={config['steps']} save_every_n_steps={config['save_every_n_steps']} rank={config['rank']} alpha={config['alpha']} lr={config['learning_rate']}",
        f"batch={config['batch_size']} grad_accum={config['gradient_accumulation']} precision={config['precision']} optimizer={config['optimizer']} memory_policy={config['memory_policy']}",
        f"dataset={config['dataset_dir']} uploaded_files={len(saved_files)} source_dir={config['source_dir'] or '-'}",
    ]
    if config.get("ltx"):
        ltx = config["ltx"]
        lines.extend(
            [
                f"ltx_training_mode={ltx['training_mode']} control_type={ltx['control_type']} dataset_mode={ltx['dataset_mode']}",
                "ltx_dataset_resolutions=" + ",".join(str(item) for item in ltx.get("dataset_resolutions") or []),
                f"ltx_caption_dropout={ltx['caption_dropout']} cache_text_embeddings={ltx['cache_text_embeddings']} weight_decay={ltx['weight_decay']}",
                f"ltx_timestep_type={ltx['timestep_type']} timestep_bias={ltx['timestep_bias']} loss_type={ltx['loss_type']}",
                f"ltx_sample_every={ltx['sample_every']} sample_sampler={ltx['sample_sampler']} sample_guidance={ltx['sample_guidance_scale']} sample_steps={ltx['sample_steps']} sample_size={ltx['sample_size']}",
                f"ltx_attention_strength={ltx['attention_strength']} target_fps={ltx['target_fps']} text_encoder_path={ltx['text_encoder_path'] or '-'}",
                "comfy_nodes=" + ",".join(ltx.get("comfy_nodes") or ["standard_ltx_lora"]),
            ]
        )
    if (config.get("perceptual") or {}).get("enabled"):
        perceptual = config["perceptual"]
        depth = perceptual["depth_consistency"]
        noise = perceptual["weight_noise"]
        lines.extend(
            [
                f"perceptual=enabled support={perceptual['support_level']} anchor={perceptual['anchor']} image_mode={perceptual['image_mode']}",
                f"perceptual_depth_weight={depth['loss_weight']} depth_model={depth['model_id']} mask_source={depth['mask_source']} loss_split={perceptual['loss_split'] or '-'}",
                f"perceptual_weight_noise={noise['enabled']} mode={noise['mode']} sigma={noise['sigma']} log_every={noise['log_every']}",
            ]
        )
    if runner.get("available"):
        lines.append("runner=available")
        lines.append("command=" + " ".join(str(part) for part in runner.get("command") or []))
    else:
        lines.append("runner=missing")
        lines.append(str(runner.get("install_hint") or "Install trainer before launch."))
    return "\n".join(lines) + "\n"


def public_train_lora_job(job: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in job.items() if not key.startswith("_")}


def train_lora_command_text(job: dict[str, Any]) -> str:
    command = ((job.get("runner") or {}).get("command") or [])
    return " ".join(str(part) for part in command)
