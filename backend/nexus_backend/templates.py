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
        "steps": 4,
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
        "steps": 4,
        "cfg": 1.0,
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
    return json.loads(TEMPLATES_PATH.read_text(encoding="utf-8"))
