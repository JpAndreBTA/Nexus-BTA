from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .config import CONFIG_DIR


TEMPLATES_PATH = CONFIG_DIR / "model_templates.json"


DEFAULT_MODEL_TEMPLATES: dict[str, dict[str, Any]] = {
    "SD": {
        "label": "SD 1.5",
        "family": "sd15",
        "type": "image",
        "model_folder": "./models/checkpoints/sd15",
        "native_size": [512, 512],
        "sampler": "euler_ancestral",
        "scheduler": "karras",
        "steps": 20,
        "cfg": 7.0,
        "supports": ["lora", "embeddings", "controlnet", "xformers", "low_vram"],
        "workflow_hint": "basic_sd_image",
    },
    "XL": {
        "label": "SDXL",
        "family": "sdxl",
        "type": "image",
        "model_folder": "./models/checkpoints/sdxl",
        "native_size": [512, 512],
        "sampler": "dpmpp_2m_sde",
        "scheduler": "karras",
        "steps": 25,
        "cfg": 6.0,
        "supports": ["lora", "refiner", "controlnet_xl", "highres"],
        "workflow_hint": "basic_sd_image",
    },
    "Flux": {
        "label": "FLUX",
        "family": "flux",
        "type": "image",
        "model_folder": "./models/checkpoints/flux",
        "native_size": [512, 512],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 20,
        "cfg": 3.5,
        "supports": ["t5", "ae_vae", "gguf", "distilled"],
        "workflow_hint": "custom_workflow",
    },
    "Qwen": {
        "label": "QWEN",
        "family": "qwen",
        "type": "image",
        "model_folder": "./models/checkpoints/qwen",
        "native_size": [512, 512],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 12,
        "cfg": 1.0,
        "supports": ["multimodal_edit", "gguf", "vl_encoder"],
        "workflow_hint": "custom_workflow",
    },
    "ZImageTurbo": {
        "label": "Z-Image Turbo",
        "family": "zimage",
        "type": "image",
        "model_folder": "./models/diffusion_models",
        "native_size": [1024, 1024],
        "sampler": "res_multistep",
        "scheduler": "simple",
        "steps": 8,
        "cfg": 1.0,
        "supports": ["z_image_turbo", "qwen3_encoder", "ae_vae", "lora"],
        "workflow_hint": "zimage_turbo",
    },
    "Lumina": {
        "label": "LUMINA",
        "family": "lumina",
        "type": "image",
        "model_folder": "./models/checkpoints/lumina",
        "native_size": [1024, 1024],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 28,
        "cfg": 4.5,
        "supports": ["t5", "flow_matching", "highres"],
        "workflow_hint": "custom_workflow",
    },
    "Wan": {
        "label": "WAN 2.2",
        "family": "wan",
        "type": "video",
        "model_folder": "./models/checkpoints/wan",
        "native_size": [512, 512],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 12,
        "cfg": 1.0,
        "denoise": 0.45,
        "supports": ["text_to_video", "image_to_video", "gguf", "high_noise", "low_noise", "4_step"],
        "workflow_hint": "custom_workflow",
    },
    "LTX": {
        "label": "LTX 2.3",
        "family": "ltx",
        "type": "video",
        "model_folder": "./models/checkpoints/ltx",
        "native_size": [512, 512],
        "sampler": "euler_cfg_pp",
        "scheduler": "quadratic",
        "steps": 8,
        "cfg": 1.0,
        "supports": ["ltx_2_3", "image_to_video", "distilled_lora_1", "distilled_lora_2"],
        "workflow_hint": "ltx23",
    },
    "LTX25": {
        "label": "LTX 2.5",
        "family": "ltx_2_5",
        "type": "video",
        "model_folder": "./models/checkpoints/ltx_25",
        "native_size": [832, 480],
        "sampler": "euler_ancestral",
        "scheduler": "simple",
        "steps": 8,
        "cfg": 1.0,
        "supports": [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "native_audio",
            "latent_spatial_upscale_x2",
            "two_stage_refiner",
            "turbo_lora",
            "fp8",
            "rtx_3060_local",
            "rtx_5090",
        ],
        "unsupported": ["ltx23_motion_transfer", "ltx23_ic_lora_video_reference"],
        "workflow_hint": "ltx25",
    },
    "MiniMaxH3": {
        "label": "MiniMax H3",
        "family": "minimax_h3",
        "type": "video",
        "model_folder": "./models/diffusion_models/minimax_h3",
        "native_size": [832, 480],
        "sampler": "res_multistep",
        "scheduler": "simple",
        "steps": 20,
        "cfg": 1.0,
        "supports": [
            "text_to_video",
            "image_to_video",
            "first_last_frame",
            "reference_to_video",
            "video_to_video_reference",
            "multi_reference_images_9",
            "reference_videos_3",
            "reference_audios_3",
            "native_audio",
            "sage_attention",
            "rtx_3060_local",
            "rtx_5090",
        ],
        "unsupported": ["motion_transfer", "pose_transfer", "depth_transfer", "mid_frame"],
        "workflow_hint": "minimax_h3",
    },
    "Krea2": {
        "label": "KREA 2",
        "family": "krea2",
        "type": "image",
        "model_folder": "./models/diffusion_models/krea2",
        "native_size": [1024, 1024],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 8,
        "cfg": 1.0,
        "supports": ["text_to_image", "style_reference", "multi_reference_images_3", "krea2_style_lora", "qwen3vl_4b", "rtx_3060_local", "rtx_5090"],
        "unsupported": ["video", "controlnet", "inpaint"],
        "workflow_hint": "krea2",
    },
    "Anima": {
        "label": "ANIMA",
        "family": "anima",
        "type": "image",
        "model_folder": "./models/checkpoints/anima",
        "native_size": [512, 512],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 16,
        "cfg": 4.0,
        "supports": ["anime_lora", "vae_auto", "controlnet", "gallery_metadata"],
        "workflow_hint": "anima",
    },
    "Ideogram4": {
        "label": "IDEOGRAM 4",
        "family": "ideogram4",
        "type": "image",
        "model_folder": "./models/diffusion_models/ideogram4",
        "native_size": [512, 512],
        "sampler": "euler",
        "scheduler": "simple",
        "steps": 4,
        "cfg": 1.0,
        "supports": ["structured_json", "regional_bbox", "qwen3vl_encoder", "flux2_vae", "low_vram_fp8"],
        "workflow_hint": "ideogram4",
    },
    "Model3D": {
        "label": "3D MODEL",
        "family": "trellis2",
        "type": "3d",
        "model_folder": "./models/3d/trellis2",
        "native_size": [1024, 1024],
        "sampler": "trellis2_image_to_3d",
        "scheduler": "projection_multiview",
        "steps": 14,
        "cfg": 8.0,
        "supports": ["image_to_3d", "multiview", "texture_paint", "controlnet_texture", "glb", "wasm_preview"],
        "workflow_hint": "model3d_trellis2",
    },
}


def ensure_templates_file() -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not TEMPLATES_PATH.exists():
        TEMPLATES_PATH.write_text(
            json.dumps(DEFAULT_MODEL_TEMPLATES, indent=2),
            encoding="utf-8",
        )


def load_templates() -> dict[str, dict[str, Any]]:
    ensure_templates_file()
    # Existing installations have a user-owned model_templates.json.  Merge
    # newly shipped presets into it instead of requiring users to delete or
    # overwrite their configuration just to receive a new template.
    saved = json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
    if not isinstance(saved, dict):
        return dict(DEFAULT_MODEL_TEMPLATES)
    merged = {key: dict(value) for key, value in DEFAULT_MODEL_TEMPLATES.items()}
    for key, value in saved.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged
