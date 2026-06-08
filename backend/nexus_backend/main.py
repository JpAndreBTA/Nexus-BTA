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
import sys
import time
import uuid
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .asset_resolver import resolve_generation_assets
from .civitai import download_civitai_asset, resolve_civitai_asset, search_civitai_models
from .comfy_client import ComfyClient, extract_outputs
from .config import DEFAULT_MODEL_SOURCES, coerce_path_list, load_settings, save_settings, sync_startup_model_path
from .dependencies import (
    custom_node_dependency_status,
    custom_node_requirements,
    custom_nodes_for_workflow,
    install_custom_node_dependencies,
    manager_suggestions_for_nodes,
)
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
    SettingsUpdate,
    WorkflowSaveRequest,
)
from .templates import ensure_templates_file, load_templates
from .workflows import (
    LTX_OMNICINE_DEFAULT_STRENGTH,
    LTX_OMNICINE_LORA_NAME,
    WorkflowRegistry,
    build_basic_anima_workflow,
    build_basic_flux_workflow,
    build_basic_ideogram4_workflow,
    build_basic_ltx_img2video_workflow,
    build_basic_qwen_image_workflow,
    build_basic_sd_workflow,
    build_basic_wan_i2video_workflow,
    build_basic_wan_motion_capture_workflow,
    build_basic_wan_video_reference_workflow,
    build_basic_zimage_turbo_workflow,
    convert_ui_to_api,
    detect_workflow_format,
    ensure_inpaint_engine_route,
    patch_workflow,
)


settings = load_settings()
generation_jobs: dict[str, dict[str, Any]] = {}
extras_jobs: dict[str, dict[str, Any]] = {}
download_jobs: dict[str, dict[str, Any]] = {}
train_lora_jobs: dict[str, dict[str, Any]] = {}
generation_lock = asyncio.Lock()
comfy_idle_task: asyncio.Task[None] | None = None
last_generation_model_signature: tuple[str, str] | None = None
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

LTX_HF_LORA_ARTIFACTS: dict[str, dict[str, str]] = {
    "control": {
        "label": "LTX 2.3 IC-LoRA Union Control",
        "filename": "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors?download=true",
    },
    "transition": {
        "label": "LTX 2.3 Transition LoRA",
        "filename": "ltx2.3-transition.safetensors",
        "url": "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors?download=true",
    },
    "detailer": {
        "label": "LTX IC-LoRA Detailer",
        "filename": "ltx-2-19b-ic-lora-detailer.safetensors",
        "url": "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors",
    },
    "cameraman": {
        "label": "LTX 2.3 IC-LoRA Cameraman",
        "filename": "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors",
        "url": "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1/resolve/main/LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors",
    },
    "outpaint": {
        "label": "LTX 2.3 IC-LoRA Outpaint",
        "filename": "ltx-2.3-22b-ic-lora-outpaint.safetensors",
        "url": "https://huggingface.co/oumoumad/LTX-2.3-22b-IC-LoRA-Outpaint/resolve/main/ltx-2.3-22b-ic-lora-outpaint.safetensors",
    },
    "denoise": {
        "label": "FastDVDnet Video Denoise",
        "filename": "fastdvdnet_model_clipped_noise.pth",
        "url": "https://raw.githubusercontent.com/m-tassano/fastdvdnet/master/model_clipped_noise.pth",
    },
    "flashvsr": {
        "label": "FlashVSR Video Upscale",
        "filename": "Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors",
        "url": "https://huggingface.co/1038lab/FlashVSR/resolve/main/Wan2_1-T2V-1_3B_FlashVSR_fp32.safetensors",
    },
    "seedvr2": {
        "label": "SeedVR2 Video Restore",
        "filename": "seedvr2_ema_3b_fp8_e4m3fn.safetensors",
        "url": "https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/seedvr2_ema_3b_fp8_e4m3fn.safetensors",
    },
    "face_restore": {
        "label": "GFPGAN Face Restoration",
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
    },
}

WAN22_HF_ARTIFACTS: dict[str, dict[str, Any]] = {
    "clip_vision": {
        "label": "WAN CLIP Vision encoder",
        "filename": "clip_vision_h.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors?download=true",
        "target": ("clip_vision", "clip_vision_h.safetensors"),
        "min_bytes": 500 * 1024 * 1024,
        "kind": "encoder",
        "scope": "dependency",
    },
    "umt5": {
        "label": "WAN UMT5 text encoder",
        "filename": "umt5_xxl_fp8_e4m3fn_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors?download=true",
        "target": ("text_encoders", "umt5_xxl_fp8_e4m3fn_scaled.safetensors"),
        "min_bytes": 1 * 1024 * 1024 * 1024,
        "kind": "encoder",
        "scope": "dependency",
    },
    "vae21": {
        "label": "WAN 2.1/2.2 video VAE",
        "filename": "wan_2.1_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/vae/wan_2.1_vae.safetensors?download=true",
        "target": ("vae", "wan_2.1_vae.safetensors"),
        "min_bytes": 100 * 1024 * 1024,
        "kind": "vae",
        "scope": "dependency",
    },
    "vae22": {
        "label": "WAN 2.2 TI2V VAE",
        "filename": "wan2.2_vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/vae/wan2.2_vae.safetensors?download=true",
        "target": ("vae", "wan2.2_vae.safetensors"),
        "min_bytes": 100 * 1024 * 1024,
        "kind": "vae",
        "scope": "dependency",
    },
    "high_noise": {
        "label": "WAN 2.2 high-noise diffusion model",
        "filename": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors?download=true",
        "target": ("diffusion_models", "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"),
        "min_bytes": 1 * 1024 * 1024 * 1024,
        "kind": "checkpoint",
        "scope": "base_model",
    },
    "low_noise": {
        "label": "WAN 2.2 low-noise diffusion model",
        "filename": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors?download=true",
        "target": ("diffusion_models", "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"),
        "min_bytes": 1 * 1024 * 1024 * 1024,
        "kind": "checkpoint",
        "scope": "base_model",
    },
}

IDEOGRAM4_HF_ARTIFACTS: dict[str, dict[str, Any]] = {
    "checkpoint": {
        "label": "Ideogram 4 FP8 model",
        "filename": "ideogram4_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_fp8_scaled.safetensors?download=true",
        "target": ("diffusion_models", "ideogram4", "ideogram4_fp8_scaled.safetensors"),
        "min_bytes": 8 * 1024 * 1024 * 1024,
        "kind": "checkpoint",
        "scope": "model",
    },
    "unconditional_checkpoint": {
        "label": "Ideogram 4 unconditional FP8 model",
        "filename": "ideogram4_unconditional_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Ideogram-4/resolve/main/diffusion_models/ideogram4_unconditional_fp8_scaled.safetensors?download=true",
        "target": ("diffusion_models", "ideogram4", "ideogram4_unconditional_fp8_scaled.safetensors"),
        "min_bytes": 8 * 1024 * 1024 * 1024,
        "kind": "checkpoint",
        "scope": "model",
    },
    "qwen3vl": {
        "label": "Ideogram 4 Qwen3-VL text encoder",
        "filename": "qwen3vl_8b_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Qwen3-VL/resolve/main/text_encoders/qwen3vl_8b_fp8_scaled.safetensors?download=true",
        "target": ("text_encoders", "qwen3vl_8b_fp8_scaled.safetensors"),
        "min_bytes": 9 * 1024 * 1024 * 1024,
        "kind": "text_encoder",
        "scope": "text_encoder",
    },
    "vae": {
        "label": "Flux2 VAE for Ideogram 4",
        "filename": "flux2-vae.safetensors",
        "url": "https://huggingface.co/Comfy-Org/flux2-dev/resolve/main/split_files/vae/flux2-vae.safetensors?download=true",
        "target": ("vae", "flux2-vae.safetensors"),
        "min_bytes": 300 * 1024 * 1024,
        "kind": "vae",
        "scope": "vae",
    },
    "gemma4": {
        "label": "Gemma 4 prompt helper encoder",
        "filename": "gemma4_e4b_it_fp8_scaled.safetensors",
        "url": "https://huggingface.co/Comfy-Org/gemma-4/resolve/main/text_encoders/gemma4_e4b_it_fp8_scaled.safetensors?download=true",
        "target": ("text_encoders", "gemma4_e4b_it_fp8_scaled.safetensors"),
        "min_bytes": 8 * 1024 * 1024 * 1024,
        "kind": "text_encoder",
        "scope": "optional_prompt_helper",
    },
}

IDEOGRAM4_REQUIRED_COMFY_NODES = (
    "ModelSamplingAuraFlow",
    "BasicScheduler",
    "ExtendIntermediateSigmas",
    "CFGOverride",
    "DualModelGuider",
)

WAN_MOTION_CAPTURE_CUSTOM_NODES: dict[str, dict[str, str]] = {
    "wan_animate_preprocess": {
        "label": "ComfyUI Wan Animate Preprocess",
        "repo": "https://github.com/kijai/ComfyUI-WanAnimatePreprocess.git",
        "folder": "ComfyUI-WanAnimatePreprocess",
        "source": "GitHub",
    },
    "wan_video_wrapper": {
        "label": "ComfyUI WanVideoWrapper",
        "repo": "https://github.com/kijai/ComfyUI-WanVideoWrapper.git",
        "folder": "ComfyUI-WanVideoWrapper",
        "source": "GitHub",
    },
}

WAN_MOTION_CAPTURE_ARTIFACTS: dict[str, dict[str, Any]] = {
    "yolo_det": {
        "label": "WAN Animate YOLO detector",
        "filename": "yolov10m.onnx",
        "url": "https://huggingface.co/Wan-AI/Wan2.2-Animate-14B/resolve/main/process_checkpoint/det/yolov10m.onnx?download=true",
        "target": ("detection", "yolov10m.onnx"),
        "min_bytes": 50 * 1024 * 1024,
        "size_bytes": 60 * 1024 * 1024,
        "source": "Hugging Face",
        "scope": "dependency",
    },
    "vitpose_l": {
        "label": "ViTPose whole-body large ONNX",
        "filename": "vitpose-l-wholebody.onnx",
        "url": "https://huggingface.co/JunkyByte/easy_ViTPose/resolve/main/onnx/wholebody/vitpose-l-wholebody.onnx?download=true",
        "target": ("detection", "vitpose-l-wholebody.onnx"),
        "min_bytes": 300 * 1024 * 1024,
        "size_bytes": 360 * 1024 * 1024,
        "source": "Hugging Face",
        "scope": "dependency",
    },
    "wan_i2v_distill_lora": {
        "label": "WAN I2V LightX2V distill LoRA",
        "filename": "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors",
        "url": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Lightx2v/lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors?download=true",
        "target": ("loras", "wan", "lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors"),
        "min_bytes": 100 * 1024 * 1024,
        "size_bytes": 260 * 1024 * 1024,
        "source": "Hugging Face",
        "scope": "dependency",
    },
    "wan_fun_5b_q4ks": {
        "label": "WAN 2.2 Fun-Control 5B GGUF Q4_K_S base model",
        "filename": "Wan2.2-Fun-5B-Control-Q4_K_S.gguf",
        "url": "https://huggingface.co/QuantStack/Wan2.2-Fun-5B-Control-GGUF/resolve/main/Wan2.2-Fun-5B-Control-Q4_K_S.gguf?download=true",
        "target": ("unet", "wan", "Wan2.2-Fun-5B-Control-Q4_K_S.gguf"),
        "min_bytes": 2 * 1024 * 1024 * 1024,
        "size_bytes": 3_130_000_000,
        "source": "Hugging Face",
        "scope": "base_model",
    },
}

WAN_MOTION_CAPTURE_NODE_GROUPS: dict[str, tuple[str, ...]] = {
    "video_loader": ("LoadVideo", "VHS_LoadVideo"),
    "image_loader": ("LoadImage",),
    "video_saver": ("SaveVideo", "VHS_VideoCombine"),
    "wan_fun_control": ("Wan22FunControlToVideo",),
    "dwpose_preprocessor": ("DWPreprocessor", "OpenposePreprocessor"),
}

POSE_QWEN_LORA_ARTIFACT: dict[str, Any] = {
    "label": "VNCCS Qwen Image Edit 2511 PoseStudio LoRA",
    "filename": "VNCCS_QIE2511_PoseStudio_ART_V5.9.5.safetensors",
    "version": "5.9.5",
    "size_bytes": 1_179_883_808,
    "repo": "MIUProject/VNCCS_PoseStudio",
    "url": "https://huggingface.co/MIUProject/VNCCS_PoseStudio/resolve/main/models/loras/qwen/VNCCS/VNCCS_QIE2511_PoseStudio_ART_V5.9.5.safetensors?download=true",
}

POSE_QWEN_CONTROLNET_ARTIFACT: dict[str, Any] = {
    "label": "Qwen Image InstantX ControlNet Union",
    "filename": "Qwen-Image-InstantX-ControlNet-Union.safetensors",
    "size_bytes": 4_281_779_224,
    "repo": "InstantX/Qwen-Image-ControlNet-Union",
    "url": "https://huggingface.co/InstantX/Qwen-Image-ControlNet-Union/resolve/main/diffusion_pytorch_model.safetensors?download=true",
}

POSE_MEDIAPIPE_LANDMARKER_ARTIFACT: dict[str, Any] = {
    "label": "MediaPipe Pose Landmarker Full",
    "filename": "pose_landmarker_full.task",
    "size_bytes": 9_398_198,
    "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}

CONTROLNET_OPTIONAL_ARTIFACTS: dict[str, dict[str, Any]] = {
    "qwen_union": {
        "label": "Qwen Image InstantX ControlNet Union",
        "preset": "Qwen",
        "types": ["canny", "depth", "openpose", "dwpose", "pose", "softedge"],
        "filename": POSE_QWEN_CONTROLNET_ARTIFACT["filename"],
        "url": POSE_QWEN_CONTROLNET_ARTIFACT["url"],
        "target": ("controlnet", "qwen", POSE_QWEN_CONTROLNET_ARTIFACT["filename"]),
        "size_bytes": POSE_QWEN_CONTROLNET_ARTIFACT["size_bytes"],
        "min_bytes": 1_000_000_000,
        "scope": "dependency",
    },
    "flux_union": {
        "label": "FLUX.1 ControlNet Union Pro 2.0",
        "preset": "Flux",
        "types": ["canny", "depth", "openpose", "dwpose", "pose", "softedge", "tile"],
        "filename": "FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors",
        "url": "https://huggingface.co/Shakker-Labs/FLUX.1-dev-ControlNet-Union-Pro-2.0/resolve/main/diffusion_pytorch_model.safetensors?download=true",
        "target": ("controlnet", "FLUX.1-dev-ControlNet-Union-Pro-2.0.safetensors"),
        "size_bytes": 4_281_779_224,
        "min_bytes": 1_000_000_000,
        "scope": "dependency",
    },
    "sd15_canny": {
        "label": "SD 1.5 ControlNet Canny",
        "preset": "SD",
        "types": ["canny"],
        "filename": "sd15_control_v11p_canny_fp16.safetensors",
        "url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_canny_fp16.safetensors?download=true",
        "target": ("controlnet", "sd15_control_v11p_canny_fp16.safetensors"),
        "size_bytes": 722_601_100,
        "min_bytes": 100_000_000,
        "scope": "dependency",
    },
    "sd15_depth": {
        "label": "SD 1.5 ControlNet Depth",
        "preset": "SD",
        "types": ["depth"],
        "filename": "sd15_control_v11f1p_depth_fp16.safetensors",
        "url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11f1p_sd15_depth_fp16.safetensors?download=true",
        "target": ("controlnet", "sd15_control_v11f1p_depth_fp16.safetensors"),
        "size_bytes": 722_000_000,
        "min_bytes": 100_000_000,
        "scope": "dependency",
    },
    "sd15_openpose": {
        "label": "SD 1.5 ControlNet OpenPose",
        "preset": "SD",
        "types": ["openpose", "dwpose", "pose"],
        "filename": "sd15_control_v11p_openpose_fp16.safetensors",
        "url": "https://huggingface.co/comfyanonymous/ControlNet-v1-1_fp16_safetensors/resolve/main/control_v11p_sd15_openpose_fp16.safetensors?download=true",
        "target": ("controlnet", "sd15_control_v11p_openpose_fp16.safetensors"),
        "size_bytes": 722_000_000,
        "min_bytes": 100_000_000,
        "scope": "dependency",
    },
    "sdxl_canny": {
        "label": "SDXL ControlNet Canny Small",
        "preset": "XL",
        "types": ["canny"],
        "filename": "sdxl_diffusers_canny_small.safetensors",
        "url": "https://huggingface.co/diffusers/controlnet-canny-sdxl-1.0-small/resolve/main/diffusion_pytorch_model.safetensors?download=true",
        "target": ("controlnet", "sdxl_diffusers_canny_small.safetensors"),
        "size_bytes": 320_237_152,
        "min_bytes": 100_000_000,
        "scope": "dependency",
    },
    "zimage_union": {
        "label": "Z-Image Turbo Fun ControlNet Union",
        "preset": "ZImageTurbo",
        "types": ["canny", "depth", "openpose", "dwpose", "pose"],
        "filename": "Z-Image-Turbo-Fun-Controlnet-Union.safetensors",
        "url": "https://huggingface.co/Comfy-Org/Z-Image-Turbo_ComfyUI/resolve/main/split_files/model_patches/Z-Image-Turbo-Fun-Controlnet-Union.safetensors?download=true",
        "target": ("model_patches", "Z-Image-Turbo-Fun-Controlnet-Union.safetensors"),
        "size_bytes": 1_200_000_000,
        "min_bytes": 100_000_000,
        "scope": "dependency",
    },
}

QWEN_MULTIANGLE_LORA_ARTIFACT: dict[str, str] = {
    "label": "Qwen Image Edit 2511 Multiple Angles LoRA",
    "filename": "qwen-image-edit-2511-multiple-angles-lora.safetensors",
    "url": "https://huggingface.co/fal/Qwen-Image-Edit-2511-Multiple-Angles-LoRA/resolve/main/qwen-image-edit-2511-multiple-angles-lora.safetensors",
    "size_bytes": "295140688",
}

TRELLIS2_REPO_ID = "microsoft/TRELLIS.2-4B"
DINOV3_REPO_ID = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_KAGGLE_HANDLE = "x1an9l1/facebook-dinov3-vitl16-pretrain-lvd1689m/transformers/default"
HF_TOKEN_PATH = settings.project_root / "config" / "huggingface_token.txt"

EXTRAS_VIDEO_RESTORE_NODES: dict[str, tuple[str, ...]] = {
    "flashvsr": ("ComfyUI-FlashVSR", "ComfyUI-FlashVSR_Ultra_Fast"),
    "seedvr2": ("ComfyUI-SeedVR2_VideoUpscaler", "seedvr2_videoupscaler"),
    "ltx_detailer": ("ComfyUI-LTXVideo",),
}

NVIDIA_EXTRAS_ENGINES: dict[str, dict[str, Any]] = {
    "nvidia_rtx": {
        "label": "NVIDIA RTX Video Super Resolution",
        "nodes": ("Nvidia_RTX_Nodes_ComfyUI",),
        "packages": ("nvvfx",),
        "model_required": False,
        "notes": "Uses NVIDIA RTX Video Super Resolution when the ComfyUI node is available; falls back to Lanczos in Extras.",
    },
    "nvidia_pid": {
        "label": "NVIDIA PiD Pixel Diffusion Decoder",
        "nodes": ("ComfyUI-PiD",),
        "packages": (
            "hydra",
            "omegaconf",
            "attrs",
            "einops",
            "loguru",
            "termcolor",
            "fvcore",
            "iopath",
            "pynvml",
            "wandb",
            "imageio",
            "cv2",
            "pandas",
            "safetensors",
            "huggingface_hub",
            "sentencepiece",
            "boto3",
            "botocore",
            "accelerate",
            "transformers",
            "diffusers",
        ),
        "model_required": True,
        "notes": "PiD is a latent decoder/upscaler, not a normal MP4 upscaler; large PiD weights remain opt-in through the node auto_download path.",
    },
}

NVIDIA_PID_REPO_ID = "nvidia/PiD"
NVIDIA_PID_SOURCE_REPO = "https://github.com/nv-tlabs/PiD.git"
NVIDIA_PID_HF_BASE = "https://huggingface.co/nvidia/PiD/resolve/main"
NVIDIA_PID_ASSET_SIZES = {
    "checkpoints/PiD_res2k_sr4x_official_flux_distill_4step/model_ema_bf16.pth": 2724842961,
    "checkpoints/PiD_res2k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth": 2725875153,
    "checkpoints/PiD_res2k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth": 2724842961,
    "checkpoints/PiD_res2kto4k_sr4x_official_flux_distill_4step/model_ema_bf16.pth": 2724842961,
    "checkpoints/PiD_res2kto4k_sr4x_official_flux2_distill_4step/model_ema_bf16.pth": 2725875153,
    "checkpoints/PiD_res2kto4k_sr4x_official_sd3_distill_4step/model_ema_bf16.pth": 2724842961,
    "checkpoints/ae.safetensors": 335304388,
    "checkpoints/flux2_ae.safetensors": 336211292,
    "checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors": 167666654,
}
NVIDIA_PID_PROFILES: dict[str, dict[str, Any]] = {
    "lowvram_zimage_2k": {
        "label": "Low VRAM Z-Image 2K",
        "backbone": "zimage",
        "checkpoint": "2k",
        "scale": 1,
        "steps": 4,
        "cfg": 1.0,
        "low_vram": True,
        "description": "Safest first PiD setup; downloads Flux-compatible 2K checkpoint and AE, then keeps media fallback unless a latent workflow is used.",
    },
    "zimage_2k_quality": {
        "label": "Z-Image 2K Quality",
        "backbone": "zimage",
        "checkpoint": "2k",
        "scale": 2,
        "steps": 4,
        "cfg": 1.0,
        "low_vram": True,
        "description": "Same checkpoint as low VRAM with a stronger PiD output scale for better GPUs.",
    },
    "flux_2k": {"label": "Flux 2K", "backbone": "flux", "checkpoint": "2k", "scale": 2, "steps": 4, "cfg": 1.0, "low_vram": True},
    "flux2_2k": {"label": "Flux2 2K", "backbone": "flux2", "checkpoint": "2k", "scale": 2, "steps": 4, "cfg": 1.0, "low_vram": False},
    "sd3_2k": {"label": "SD3 2K", "backbone": "sd3", "checkpoint": "2k", "scale": 2, "steps": 4, "cfg": 1.0, "low_vram": False},
    "zimage_2kto4k": {"label": "Z-Image 2K to 4K", "backbone": "zimage", "checkpoint": "2kto4k", "scale": 4, "steps": 4, "cfg": 1.0, "low_vram": False},
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
        if update.get("prompt_id"):
            job["prompt_id"] = update.get("prompt_id")
            job["updated_at"] = datetime.now().isoformat(timespec="seconds")
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


def _generation_job_cancelled(job_id: str | None) -> bool:
    return bool(job_id and str(generation_jobs.get(job_id, {}).get("status") or "").lower() == "cancelled")


def _raise_if_generation_cancelled(job_id: str | None) -> None:
    if _generation_job_cancelled(job_id):
        raise RuntimeError("Generation cancelled.")


def _handle_cancelled_generation_progress(job_id: str | None, update: dict[str, Any]) -> bool:
    if not _generation_job_cancelled(job_id):
        return False
    prompt_id = update.get("prompt_id")
    if prompt_id and job_id and generation_jobs.get(job_id):
        generation_jobs[job_id]["prompt_id"] = prompt_id
        generation_jobs[job_id]["updated_at"] = datetime.now().isoformat(timespec="seconds")
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(comfy.interrupt(str(prompt_id)))
            loop.create_task(comfy.clear_queue())
        except RuntimeError:
            pass
    return True


def _public_generation_job(job: dict[str, Any]) -> dict[str, Any]:
    if str(job.get("status") or "").lower() == "completed" and not job.get("outputs"):
        try:
            created = datetime.fromisoformat(str(job.get("created_at") or ""))
            start_timestamp = created.timestamp()
        except Exception:
            start_timestamp = datetime.now().timestamp() - 600
        recovered_outputs = _cleanup_video_sidecar_images(_recent_output_files(start_timestamp - 300, limit=20), start_timestamp)
        if recovered_outputs:
            job["outputs"] = recovered_outputs
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


def _download_relative_path(path: Path) -> str:
    try:
        return str(path.relative_to(settings.project_root))
    except ValueError:
        try:
            return str(path.relative_to(settings.models_dir))
        except ValueError:
            return str(path)


def _write_input_data_image(value: str, prefix: str, *, normalize: bool = True) -> str:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    match = re.match(r"data:image/([a-zA-Z0-9.+-]+);base64,(.+)", value, flags=re.DOTALL)
    if not match:
        raise ValueError("Invalid image data URL.")
    ext = "jpg" if match.group(1).lower() in {"jpeg", "jpg"} else "png"
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.{ext}"
    target = settings.input_dir / filename
    target.write_bytes(base64.b64decode(match.group(2)))
    if normalize:
        _normalize_input_image_file(target)
    return filename


def _normalize_input_image_file(path: Path) -> None:
    if path.suffix.lower() != ".png":
        return
    try:
        from PIL import Image

        with Image.open(path) as image:
            bands = image.getbands()
            alpha_like = "A" in bands or image.mode == "LA" or (image.mode == "P" and "transparency" in image.info)
            target_mode = "RGBA" if alpha_like else "RGB"
            if not alpha_like and image.mode == target_mode:
                return
            converted = image.convert(target_mode)
            if alpha_like:
                alpha = converted.getchannel("A")
                alpha_min, alpha_max = alpha.getextrema()
                if alpha_min == 255 and alpha_max == 255:
                    try:
                        import numpy as np

                        rgba = np.array(converted)
                        rgb = rgba[:, :, :3].astype(np.int16)
                        border = np.concatenate(
                            [
                                rgb[:3, :, :].reshape(-1, 3),
                                rgb[-3:, :, :].reshape(-1, 3),
                                rgb[:, :3, :].reshape(-1, 3),
                                rgb[:, -3:, :].reshape(-1, 3),
                            ],
                            axis=0,
                        )
                        background = np.median(border, axis=0).astype(np.int16)
                        candidate = np.max(np.abs(rgb - background), axis=2) <= 36
                        visited = np.zeros(candidate.shape, dtype=bool)
                        stack: list[tuple[int, int]] = []
                        height, width = candidate.shape
                        for x in range(width):
                            if candidate[0, x]:
                                stack.append((0, x))
                            if candidate[height - 1, x]:
                                stack.append((height - 1, x))
                        for y in range(height):
                            if candidate[y, 0]:
                                stack.append((y, 0))
                            if candidate[y, width - 1]:
                                stack.append((y, width - 1))
                        while stack:
                            y, x = stack.pop()
                            if y < 0 or x < 0 or y >= height or x >= width or visited[y, x] or not candidate[y, x]:
                                continue
                            visited[y, x] = True
                            stack.append((y - 1, x))
                            stack.append((y + 1, x))
                            stack.append((y, x - 1))
                            stack.append((y, x + 1))
                        if visited.any() and visited.mean() > 0.05:
                            cleaned_alpha = Image.fromarray(np.where(visited, 0, 255).astype("uint8"), mode="L")
                        else:
                            cleaned_alpha = alpha
                    except Exception:
                        cleaned_alpha = alpha
                else:
                    cleaned_alpha = alpha.point(lambda value: 0 if value < 48 else 255)
                bbox = cleaned_alpha.getbbox()
                if bbox:
                    pad = 8
                    left = max(0, bbox[0] - pad)
                    top = max(0, bbox[1] - pad)
                    right = min(converted.width, bbox[2] + pad)
                    bottom = min(converted.height, bbox[3] + pad)
                    crop_box = (left, top, right, bottom)
                    converted = converted.crop(crop_box)
                    cleaned_alpha = cleaned_alpha.crop(crop_box)
                    converted.putalpha(cleaned_alpha)
                else:
                    converted.putalpha(cleaned_alpha)
            converted.save(path, format="PNG")
    except Exception:
        return


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
    video_replacements: dict[str, str] = {}
    missing_custom_audio: list[str] = []

    def remember_video_replacement(old_value: object, filename: str) -> None:
        if not old_value or not filename:
            return
        value = str(old_value).strip()
        if not value:
            return
        video_replacements[value.lower()] = filename
        video_replacements[Path(value).name.lower()] = filename

    def materialize_motion_entry(entry: Any, prefix: str) -> bool:
        if not isinstance(entry, dict):
            return False
        video_b64 = str(entry.get("videoB64") or entry.get("videoSrc") or "").strip()
        if not video_b64.startswith("data:video/"):
            return False
        old_values = [
            entry.get("videoFile"),
            entry.get("video"),
            entry.get("fileName"),
            video_b64,
        ]
        filename = _write_input_data_video(video_b64, prefix)
        entry["videoFile"] = filename
        entry["video"] = filename
        entry["fileName"] = entry.get("fileName") or filename
        entry["videoB64"] = ""
        entry["videoSrc"] = ""
        for old_value in old_values:
            remember_video_replacement(old_value, filename)
        return True

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
                    motion = segment.get("motionTransfer")
                    if materialize_motion_entry(motion, "nexus_director_motion"):
                        changed = True
                    continue
                old_video_values = [
                    segment.get("videoFile"),
                    segment.get("video"),
                    segment.get("fileName"),
                    video_b64,
                ]
                filename = _write_input_data_video(video_b64, "nexus_director_video")
                segment["videoFile"] = filename
                segment["fileName"] = segment.get("fileName") or filename
                segment["videoB64"] = ""
                load_video = segment.setdefault("loadVideo", {})
                if isinstance(load_video, dict):
                    load_video["video"] = filename
                changed = True
                for old_value in old_video_values:
                    remember_video_replacement(old_value, filename)

                motion = segment.get("motionTransfer")
                if materialize_motion_entry(motion, "nexus_director_motion"):
                    changed = True

        motion_entries = timeline.get("motionTransfer")
        if isinstance(motion_entries, list):
            for entry in motion_entries:
                if materialize_motion_entry(entry, "nexus_director_motion"):
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
            source_video_b64 = str(segment.get("sourceVideoB64") or "")
            old_names = [
                segment.get("audioFile"),
                segment.get("fileName"),
                segment.get("title"),
            ]
            filename = ""
            if source_video_b64.startswith("data:video/"):
                video_name = _write_input_data_video(source_video_b64, "nexus_director_audio_source")
                filename = _extract_audio_to_input(settings.input_dir / video_name, "nexus_director_audio")
                old_names.extend([segment.get("sourceVideoFile"), video_name])
                segment["sourceVideoB64"] = ""
                segment["sourceVideoFile"] = video_name
            elif audio_b64.startswith("data:audio/"):
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
                segment["sourceVideoB64"] = ""
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
        fallback = None
    else:
        fallback = next(iter(replacements.values())) if len(set(replacements.values())) == 1 else None
    for node in prompt.values():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "loadaudioui":
            if isinstance(node, dict):
                inputs = node.setdefault("inputs", {})
                class_type = str(node.get("class_type", "")).lower()
                video_key = "file" if class_type == "loadvideo" else ("video" if class_type == "vhs_loadvideo" else "")
                if video_key:
                    current_video = str(inputs.get(video_key) or "").strip()
                    if current_video.startswith("data:video/"):
                        inputs[video_key] = _write_input_data_video(current_video, "nexus_director_motion")
                    else:
                        replacement_video = video_replacements.get(current_video.lower()) or video_replacements.get(Path(current_video).name.lower())
                        if replacement_video:
                            inputs[video_key] = replacement_video
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
    return "qwen" in text and "lightning" in text


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
        if auto_lightning is not False:
            item.strength = 1.0
        cleaned.append(item)
        if _normalize_lora_key(item_name) == normalized:
            has_edit_lightning = True
    if auto_lightning is not False and not has_edit_lightning:
        cleaned.insert(0, DistilledLoraSelection(name=name, strength=1.0))
    request.distilled_loras = cleaned[:1]


def _is_qwen_multiangle_lora_name(value: object) -> bool:
    text = str(value or "").lower()
    return "qwen" in text and any(token in text for token in ("multiangle", "multi-angle", "multiple-angle", "multiple-angles", "angles-lora"))


def _ensure_qwen_multiangle_lora(request: GenerateRequest, assets: dict[str, str]) -> None:
    if request.preset.lower() != "qwen" or request.activity != "img2img":
        return
    video_options = request.video or {}
    enabled = video_options.get("qwen_multiview", False)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"", "false", "0", "off", "none", "no"}
    if not enabled:
        return
    name = assets.get("qwen_multiangle_lora")
    if not name:
        return
    normalized = _normalize_lora_key(name)
    existing = {
        _normalize_lora_key(item.get("relative_name") or item.get("relative_path") or item.get("lora_name") or item.get("name"))
        for item in request.loras
        if isinstance(item, dict)
    }
    existing.update(_normalize_lora_key(getattr(item, "name", "")) for item in request.distilled_loras)
    if normalized in existing:
        return
    strength = video_options.get("qwen_multiangle_lora_strength", 1.0)
    try:
        strength_value = float(strength)
    except (TypeError, ValueError):
        strength_value = 1.0
    request.loras.append({"name": name, "relative_name": name, "strength": strength_value, "strength_model": strength_value})


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
    _normalize_input_image_file(target)
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


def _ltx_outpaint_route_requested(request: GenerateRequest) -> bool:
    if request.preset.lower() != "ltx" or request.workflow_id != "ltx23-video-outpainting":
        return False
    video_options = request.video if isinstance(request.video, dict) else {}
    return bool(
        request.activity == "img2img"
        and getattr(request, "workspace", "") == "canvas"
        and video_options.get("outpaint_enabled")
        and (request.img2img.base_video or "").strip()
    )


def _normalize_ltx_outpaint_workflow_scope(request: GenerateRequest) -> None:
    if request.preset.lower() != "ltx" or request.workflow_id != "ltx23-video-outpainting":
        if request.preset.lower() == "ltx" and request.workflow_override and not _ltx_outpaint_route_requested(request):
            override_text = json.dumps(request.workflow_override, ensure_ascii=False).lower()
            if any(token in override_text for token in ("imagepadkj", "outpaint", "ltxaddvideoicloraguideadvanced")):
                request.workflow_override = None
                if isinstance(request.video, dict):
                    request.video["outpaint_enabled"] = False
        return
    if _ltx_outpaint_route_requested(request):
        return
    request.workflow_id = None
    request.workflow_override = None
    if isinstance(request.video, dict):
        request.video["outpaint_enabled"] = False


def _extract_video_first_frame(video_name: str, prefix: str = "nexus_base_video_frame") -> str | None:
    source = settings.input_dir / video_name
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg or not source.exists():
        return None
    target = settings.input_dir / f"{prefix}_{uuid.uuid4().hex[:10]}.png"
    try:
        _run_ffmpeg([
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-frames:v",
            "1",
            str(target),
        ])
    except Exception:
        return None
    return target.name if target.exists() else None


def _prepare_ltx_motion_scaffold(reference_names: list[str], request: GenerateRequest) -> str | None:
    if request.preset.lower() != "ltx" or len(reference_names) < 2:
        return None
    video_options = request.video or {}
    start_end_mode = str(video_options.get("ltx_start_end_mode") or "flf_guides").lower()
    if start_end_mode != "motion_scaffold":
        return None
    try:
        import cv2
        import numpy as np
        from PIL import Image, ImageOps
    except Exception:
        return None

    def load_rgb(name: str) -> "np.ndarray":
        image = Image.open(settings.input_dir / name)
        image = ImageOps.exif_transpose(image).convert("RGB")
        image = image.resize((max(64, int(request.width)), max(64, int(request.height))), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.uint8)

    try:
        start = load_rgb(reference_names[0])
        end = load_rgb(reference_names[1])
        frames = max(9, int(round(float(video_options.get("frames") or 17))))
        if (frames - 1) % 8 != 0:
            frames = (((frames - 1) // 8) + 1) * 8 + 1
        fps = max(1, int(float(video_options.get("fps") or 8)))
        start_gray = cv2.cvtColor(start, cv2.COLOR_RGB2GRAY)
        end_gray = cv2.cvtColor(end, cv2.COLOR_RGB2GRAY)
        flow_fw = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(start_gray, end_gray, None)
        flow_bw = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM).calc(end_gray, start_gray, None)
        height, width = start.shape[:2]
        grid_x, grid_y = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
        out_name = f"nexus_base_video_ltx_scaffold_{uuid.uuid4().hex[:10]}.mp4"
        target = settings.input_dir / out_name
        writer = cv2.VideoWriter(str(target), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (width, height))
        for index in range(frames):
            t = index / max(frames - 1, 1)
            eased = t * t * (3.0 - 2.0 * t)
            s_map_x = grid_x + flow_fw[..., 0] * eased
            s_map_y = grid_y + flow_fw[..., 1] * eased
            e_map_x = grid_x + flow_bw[..., 0] * (1.0 - eased)
            e_map_y = grid_y + flow_bw[..., 1] * (1.0 - eased)
            start_warp = cv2.remap(start, s_map_x, s_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            end_warp = cv2.remap(end, e_map_x, e_map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
            frame = cv2.addWeighted(start_warp, 1.0 - eased, end_warp, eased, 0.0)
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()
        video_options["ltx_motion_scaffold"] = True
        video_options["frames"] = frames
        video_options["fps"] = fps
        request.video = video_options
        return out_name
    except Exception:
        return None





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
    preset = request.preset.lower()
    raw_model = f"{request.model_name or ''} {request.model_path or ''} {request.template or ''}"
    flux2_refs = preset == "flux" and any(token in raw_model.lower() for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein"))
    max_refs = 5 if flux2_refs else (4 if preset == "model3d" else 3)
    values = _reference_image_values(request)[:max_refs]
    return [_prepare_reference_value(value, f"nexus_reference_{index + 1}") for index, value in enumerate(values)]


def _prepare_ltx_director_frame_guides(request: GenerateRequest) -> list[dict[str, Any]]:
    if request.preset.lower() != "ltx" or getattr(request, "workspace", "") != "director":
        return []
    director = request.director if isinstance(request.director, dict) else {}
    timeline = director.get("timeline_data") if isinstance(director.get("timeline_data"), dict) else {}
    if not timeline and isinstance(director.get("timeline_data_json"), str):
        try:
            parsed = json.loads(str(director.get("timeline_data_json") or "{}"))
            if isinstance(parsed, dict):
                timeline = parsed
        except Exception:
            timeline = {}
    segments_all = [segment for segment in timeline.get("segments", []) or [] if isinstance(segment, dict)]
    if any(str(segment.get("sourceType") or segment.get("type") or "").lower() in {"text", "video"} for segment in segments_all):
        return []
    if timeline.get("audioSegments"):
        return []
    if director.get("use_custom_audio"):
        return []
    fps = int(_number_or_none((request.video or {}).get("fps") or director.get("frame_rate")) or 8)
    source_fps = int(_number_or_none(director.get("frame_rate")) or fps)
    segments = [
        segment
        for segment in segments_all
        if isinstance(segment, dict) and str(segment.get("sourceType") or segment.get("type") or "image").lower() == "image"
    ]
    segments.sort(key=lambda item: int(item.get("start") or 0))
    guides: list[dict[str, Any]] = []
    for index, segment in enumerate(segments[:8]):
        image_value = str(segment.get("imageB64") or segment.get("imageSrc") or "").strip()
        if not image_value.startswith("data:image/"):
            continue
        image_name = _write_input_data_image(image_value, f"nexus_director_frame_{index + 1}")
        raw_start = int(_number_or_none(segment.get("start")) or 0)
        guide_index = raw_start
        if source_fps > 0 and fps > 0:
            guide_index = int(round(raw_start * (fps / source_fps)))
        guides.append(
            {
                "image": image_name,
                "index": -1 if index == len(segments[:8]) - 1 and len(segments[:8]) > 1 else max(0, guide_index),
                "strength": max(0.0, min(1.0, float(_number_or_none(segment.get("guideStrength")) or 1.0))),
            }
        )
    if guides and director.get("local_prompts") and not (request.prompt or "").strip():
        request.prompt = str(director.get("local_prompts") or "")
    return guides


def _prepare_ltx_director_motion_transfer(request: GenerateRequest) -> str | None:
    if request.preset.lower() != "ltx" or getattr(request, "workspace", "") != "director":
        return None
    director = request.director or {}
    raw_timeline = director.get("timeline_data")
    if not raw_timeline and director.get("timeline_data_json"):
        try:
            raw_timeline = json.loads(str(director.get("timeline_data_json") or "{}"))
        except Exception:
            raw_timeline = {}
    if not isinstance(raw_timeline, dict):
        return None
    entries: list[dict[str, Any]] = []
    for item in raw_timeline.get("motionTransfer") or []:
        if isinstance(item, dict) and item.get("enabled"):
            entries.append(item)
    for segment in raw_timeline.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        motion = segment.get("motionTransfer")
        if isinstance(motion, dict) and motion.get("enabled"):
            merged = dict(motion)
            merged.setdefault("start", segment.get("start", 0))
            merged.setdefault("duration", segment.get("length") or segment.get("duration") or 0)
            entries.append(merged)
    if not entries:
        return None
    entry = next((item for item in entries if str(item.get("videoB64") or item.get("videoSrc") or "").strip()), entries[0])
    video_value = str(entry.get("videoB64") or entry.get("videoSrc") or "").strip()
    if not video_value:
        return None
    base_video = _prepare_video_value(video_value, "nexus_ltx_director_motion")
    video_options = dict(request.video or {})
    mode = str(entry.get("mode") or entry.get("control_mode") or video_options.get("motion_transfer_control_mode") or "pose").strip().lower()
    video_options.update(
        {
            "motion_transfer_enabled": True,
            "motion_transfer_mode": "ltx_ic_union",
            "motion_transfer_control_mode": mode if mode in {"pose", "canny", "depth", "camera"} else "pose",
            "motion_transfer_motion_strength": 1,
            "motion_transfer_target_strength": float(entry.get("strength") or 1),
            "ltx_ic_lora_strength": float(entry.get("strength") or 1),
            "director_motion_transfer_segments": len(entries),
            "director_motion_transfer_source": entry.get("video") or entry.get("videoFile") or "segment video reference",
        }
    )
    if entry.get("sourceDuration") and not video_options.get("seconds"):
        video_options["seconds"] = float(entry.get("sourceDuration") or 0)
    request.video = video_options
    request.img2img.base_video = str(settings.input_dir / base_video)
    return base_video


def _ltx_director_timeline(request: GenerateRequest) -> dict[str, Any]:
    director = request.director if isinstance(request.director, dict) else {}
    timeline = director.get("timeline_data") if isinstance(director.get("timeline_data"), dict) else {}
    if not timeline and isinstance(director.get("timeline_data_json"), str):
        try:
            parsed = json.loads(str(director.get("timeline_data_json") or "{}"))
            if isinstance(parsed, dict):
                timeline = parsed
        except Exception:
            timeline = {}
    return timeline if isinstance(timeline, dict) else {}


def _ltx_director_segment_render_requested(request: GenerateRequest) -> bool:
    if request.preset.lower() != "ltx" or getattr(request, "workspace", "") != "director":
        return False
    video_options = request.video or {}
    if not video_options.get("director_segment_render"):
        return False
    timeline = _ltx_director_timeline(request)
    for segment in timeline.get("segments") or []:
        if not isinstance(segment, dict):
            continue
        motion = segment.get("motionTransfer") if isinstance(segment.get("motionTransfer"), dict) else {}
        if motion.get("enabled") or segment.get("transitionLoraEnabled"):
            return True
    return False


def _clone_generate_request(request: GenerateRequest) -> GenerateRequest:
    if hasattr(request, "model_copy"):
        return request.model_copy(deep=True)  # type: ignore[attr-defined]
    return request.copy(deep=True)  # type: ignore[no-any-return]


def _director_segment_seconds(segment: dict[str, Any], fps: float, fallback: float = 2.0) -> float:
    raw = _number_or_none(segment.get("length") or segment.get("duration"))
    if raw is None:
        return fallback
    if raw > 24:
        return max(0.25, float(raw) / max(1.0, fps))
    return max(0.25, float(raw))


def _director_segment_frame_count(seconds: float, fps: float) -> int:
    frames = max(9, int(round(max(0.25, seconds) * max(1.0, fps))) + 1)
    if (frames - 1) % 8 != 0:
        frames = (((frames - 1) // 8) + 1) * 8 + 1
    return frames


def _director_materialize_image_value(value: str, prefix: str) -> str | None:
    value = str(value or "").strip()
    if not value:
        return None
    if value.startswith("data:image/") or value.startswith("/outputs/") or "/outputs/" in value or Path(value).exists():
        return _prepare_reference_value(value, prefix)
    candidate = (settings.input_dir / value).resolve()
    try:
        if candidate.exists() and candidate.is_relative_to(settings.input_dir.resolve()):
            return candidate.name
    except Exception:
        pass
    return None


def _director_materialize_segment_image(segment: dict[str, Any], prefix: str) -> str | None:
    for key in ("imageB64", "imageSrc", "imageFile"):
        resolved = _director_materialize_image_value(str(segment.get(key) or ""), prefix)
        if resolved:
            return resolved
    return None


def _director_materialize_motion_video(motion: dict[str, Any], prefix: str) -> str | None:
    for key in ("videoB64", "videoSrc", "videoFile", "video"):
        value = str(motion.get(key) or "").strip()
        if not value:
            continue
        if value.startswith("data:video/") or value.startswith("/outputs/") or "/outputs/" in value or Path(value).exists():
            return _prepare_video_value(value, prefix)
        candidate = (settings.input_dir / value).resolve()
        try:
            if candidate.exists() and candidate.is_relative_to(settings.input_dir.resolve()):
                return candidate.name
        except Exception:
            continue
    return None


def _normalize_director_motion_reference(video_name: str, seconds: float, fps: float, width: int, height: int, frames: int) -> str:
    source = (settings.input_dir / video_name).resolve()
    if not source.exists():
        raise ValueError("Director Motion Transfer reference video could not be resolved.")
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        return video_name
    source_duration = max(0.001, _ffprobe_duration(source))
    ratio = max(0.05, min(20.0, float(seconds) / source_duration))
    out_name = f"nexus_director_motion_sync_{uuid.uuid4().hex[:10]}.mp4"
    target = settings.input_dir / out_name
    safe_width = max(64, int(width)) - (max(64, int(width)) % 2)
    safe_height = max(64, int(height)) - (max(64, int(height)) % 2)
    vf = (
        f"setpts={ratio:.8f}*PTS,"
        f"fps={max(1.0, float(fps)):.6f},"
        f"scale={safe_width}:{safe_height}:force_original_aspect_ratio=increase,"
        f"crop={safe_width}:{safe_height},"
        "tpad=stop_mode=clone:stop_duration=10,"
        f"trim=end_frame={max(1, int(frames))},"
        f"setpts=N/({max(1.0, float(fps)):.6f}*TB),"
        "format=yuv420p"
    )
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source),
            "-an",
            "-vf",
            vf,
            "-frames:v",
            str(max(1, int(frames))),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    return out_name


def _output_path_from_item(item: dict[str, Any]) -> Path | None:
    relative = str(item.get("path") or "").strip()
    if relative:
        path = (settings.output_dir / relative).resolve()
        try:
            if path.exists() and path.is_relative_to(settings.output_dir.resolve()):
                return path
        except Exception:
            return None
    filename = str(item.get("filename") or "").strip()
    subfolder = str(item.get("subfolder") or "").strip()
    if filename:
        path = (settings.output_dir / subfolder / filename).resolve()
        try:
            if path.exists() and path.is_relative_to(settings.output_dir.resolve()):
                return path
        except Exception:
            return None
    return None


def _archive_director_segment_videos(paths: list[Path], run_stamp: str) -> list[Path]:
    segments_dir = settings.output_dir / "director" / run_stamp / "segments"
    segments_dir.mkdir(parents=True, exist_ok=True)
    archived: list[Path] = []
    for index, source in enumerate(paths, start=1):
        suffix = source.suffix.lower() or ".mp4"
        target = segments_dir / f"segment_{index:03d}_{source.stem}{suffix}"
        shutil.copy2(source, target)
        metadata = source.with_name(source.name + ".nexus.json")
        if metadata.exists():
            shutil.copy2(metadata, target.with_name(target.name + ".nexus.json"))
        archived.append(target)
    return archived


def _concat_director_segment_videos(paths: list[Path], fps: float, width: int, height: int, run_stamp: str | None = None) -> Path:
    if not paths:
        raise ValueError("Director segment render did not produce any videos.")
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        if len(paths) == 1:
            output_dir = settings.output_dir / "videos"
            output_dir.mkdir(parents=True, exist_ok=True)
            target = output_dir / f"{run_stamp or datetime.now().strftime('%Y%m%d_%H%M%S')}_LTX_DIRECTOR_SEGMENTS_{uuid.uuid4().hex[:6]}.mp4"
            shutil.copy2(paths[0], target)
            return target
        raise ValueError("FFmpeg is required to join Director segment renders.")
    work_dir = settings.temp_dir / "ltx_director_segments" / uuid.uuid4().hex[:12]
    work_dir.mkdir(parents=True, exist_ok=True)
    list_file = work_dir / "segments.txt"
    list_lines = [f"file '{path.as_posix().replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'" for path in paths]
    list_file.write_text("\n".join(list_lines), encoding="utf-8")
    output_dir = settings.output_dir / "videos"
    output_dir.mkdir(parents=True, exist_ok=True)
    target = output_dir / f"{run_stamp or datetime.now().strftime('%Y%m%d_%H%M%S')}_LTX_DIRECTOR_SEGMENTS_{uuid.uuid4().hex[:6]}.mp4"
    vf = (
        f"fps={max(1.0, float(fps)):.6f},"
        f"scale={int(width)}:{int(height)}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={int(width)}:{int(height)},format=yuv420p"
    )
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(list_file),
            "-vf",
            vf,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-preset",
            "medium",
            "-movflags",
            "+faststart",
            str(target),
        ]
    )
    return target


def _director_materialize_audio_path(segment: dict[str, Any]) -> Path | None:
    audio_b64 = str(segment.get("audioB64") or "").strip()
    source_video_b64 = str(segment.get("sourceVideoB64") or "").strip()
    if source_video_b64.startswith("data:video/"):
        video_name = _write_input_data_video(source_video_b64, "nexus_director_audio_source")
        audio_name = _extract_audio_to_input(settings.input_dir / video_name, "nexus_director_audio")
        return settings.input_dir / audio_name
    if audio_b64.startswith("data:audio/"):
        return settings.input_dir / _write_input_data_audio(audio_b64, "nexus_director_audio")
    for key in ("audioFile", "fileName", "sourceVideoFile"):
        value = str(segment.get(key) or "").strip()
        if not value:
            continue
        candidate = (settings.input_dir / Path(value).name).resolve()
        try:
            if candidate.exists() and candidate.is_relative_to(settings.input_dir.resolve()):
                if key == "sourceVideoFile" or candidate.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
                    audio_name = _extract_audio_to_input(candidate, "nexus_director_audio")
                    return settings.input_dir / audio_name
                return candidate
        except Exception:
            continue
    return None


def _mux_director_audio(final_video: Path, timeline: dict[str, Any], run_stamp: str) -> Path:
    audio_segments = [item for item in timeline.get("audioSegments") or [] if isinstance(item, dict)]
    if not audio_segments:
        return final_video
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        return final_video
    audio_inputs: list[tuple[Path, float, float]] = []
    for segment in audio_segments:
        path = _director_materialize_audio_path(segment)
        if not path:
            continue
        start = max(0.0, float(_number_or_none(segment.get("start")) or 0.0))
        length = max(0.05, float(_number_or_none(segment.get("length") or segment.get("sourceDuration")) or 0.0))
        audio_inputs.append((path, start, length))
    if not audio_inputs:
        return final_video
    temp_audio = settings.temp_dir / "ltx_director_segments" / f"{run_stamp}_audio.wav"
    temp_audio.parent.mkdir(parents=True, exist_ok=True)
    command = [ffmpeg, "-y"]
    filters: list[str] = []
    mix_labels: list[str] = []
    for index, (path, start, length) in enumerate(audio_inputs):
        command.extend(["-i", str(path)])
        delay_ms = int(round(start * 1000))
        trim = f",atrim=duration={length:.6f}" if length > 0 else ""
        label = f"a{index}"
        filters.append(f"[{index}:a]aresample=48000{trim},adelay={delay_ms}|{delay_ms}[{label}]")
        mix_labels.append(f"[{label}]")
    filters.append("".join(mix_labels) + f"amix=inputs={len(mix_labels)}:duration=longest:dropout_transition=0[aout]")
    _run_ffmpeg([*command, "-filter_complex", ";".join(filters), "-map", "[aout]", "-c:a", "pcm_s16le", str(temp_audio)])
    muxed = final_video.with_name(final_video.stem + "_audio.mp4")
    _run_ffmpeg(
        [
            ffmpeg,
            "-y",
            "-i",
            str(final_video),
            "-i",
            str(temp_audio),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-shortest",
            str(muxed),
        ]
    )
    try:
        final_video.unlink()
        muxed.replace(final_video)
    except Exception:
        return muxed
    return final_video


async def _run_ltx_director_segment_render(
    request: GenerateRequest,
    assets: dict[str, str],
    object_info: dict[str, Any],
    job_id: str | None = None,
) -> GenerateResponse | None:
    if not _ltx_director_segment_render_requested(request):
        return None
    timeline = _ltx_director_timeline(request)
    all_segments = [segment for segment in timeline.get("segments") or [] if isinstance(segment, dict)]
    visual_segments = [
        segment
        for segment in all_segments
        if str(segment.get("sourceType") or segment.get("type") or "image").lower() in {"image", "video", "text"}
    ]
    if not visual_segments:
        return None
    visual_segments.sort(key=lambda item: float(_number_or_none(item.get("start")) or 0))
    fps = max(1.0, float(_number_or_none((request.video or {}).get("fps") or request.director.get("frame_rate")) or 24))
    width = max(64, int(request.width)) - (max(64, int(request.width)) % 32)
    height = max(64, int(request.height)) - (max(64, int(request.height)) % 32)
    checkpoint_name = assets.get("primary_model") or Path(request.model_path or request.model_name or "").name
    text_encoder_name = assets.get("text_encoder")
    if not checkpoint_name or not text_encoder_name:
        raise ValueError("LTX Director segment render requires an LTX checkpoint and Gemma text encoder.")
    if not assets.get("text_projection") or not (assets.get("video_vae") or assets.get("vae")) or not assets.get("audio_vae"):
        raise ValueError("LTX Director segment render requires LTX 2.3 text projection, video VAE and audio VAE.")
    available_nodes = set(object_info or {})
    output_paths: list[Path] = []
    last_prompt_id: str | None = None
    director_run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for index, segment in enumerate(visual_segments):
        _raise_if_generation_cancelled(job_id)
        motion = segment.get("motionTransfer") if isinstance(segment.get("motionTransfer"), dict) else {}
        motion_mode = str(motion.get("mode") or "pose").strip().lower()
        if motion_mode not in {"pose", "canny", "depth", "camera"}:
            motion_mode = "pose"
        segment_width = 768 if motion.get("enabled") and motion_mode == "camera" else width
        segment_height = 512 if motion.get("enabled") and motion_mode == "camera" else height
        segment_request = _clone_generate_request(request)
        segment_request.activity = "img2img"
        segment_request.workspace = "director_segment"
        segment_request.workflow_override = None
        segment_request.workflow_id = None
        segment_request.width = segment_width
        segment_request.height = segment_height
        segment_request.cfg = float(request.cfg or 1.0)
        segment_request.steps = int(request.steps or 4)
        segment_request.sampler = request.sampler or "euler_cfg_pp"
        segment_prompt = str(segment.get("prompt") or request.prompt or "").strip()
        segment_request.prompt = segment_prompt or request.prompt or "preserve the same subject identity, natural continuous motion"
        segment_request.negative_prompt = str(segment.get("negativePrompt") or request.negative_prompt or "").strip()
        if motion.get("enabled") and motion_mode == "camera":
            camera_identity_prompt = (
                "preserve the exact target subject identity, same face, same facial proportions, "
                "same outfit and lighting, apply the reference camera motion clearly: slow cinematic camera orbit "
                "and horizontal pan around the target scene, visible parallax and camera rotation while preserving "
                "the target subject and setting, continuous stable video"
            )
            segment_request.prompt = f"{segment_request.prompt}, {camera_identity_prompt}" if segment_request.prompt else camera_identity_prompt
        if segment_request.negative_prompt:
            segment_request.negative_prompt += ", changed identity, different person, ghost face, smear, crossfade, dissolve"
        else:
            segment_request.negative_prompt = "changed identity, different person, ghost face, smear, crossfade, dissolve"
        if motion.get("enabled") and motion_mode == "camera":
            segment_request.negative_prompt += (
                ", different person, changed face, changed ethnicity, asian facial features, east asian face, "
                "kpop face, identity drift, double face, ghost face, smear, excessive blur, frozen frame, "
                "control map texture"
            )
        seconds = _director_segment_seconds(segment, fps, fallback=max(2.0, float(_number_or_none((request.video or {}).get("seconds")) or 2.0)))
        frames = _director_segment_frame_count(seconds, fps)
        segment_request.video = {
            **(request.video or {}),
            "fps": fps,
            "seconds": seconds,
            "duration": seconds,
            "frames": frames,
            "length": frames,
            "active_audio": False,
            "director_segment_render": False,
            "motion_transfer_enabled": False,
            "transition_lora_enabled": False,
        }
        reference_image_name = _director_materialize_segment_image(segment, f"nexus_director_segment_{index + 1}")
        if not reference_image_name:
            next_image = next(
                (
                    name
                    for item in visual_segments
                    if item is not segment
                    for name in [_director_materialize_segment_image(item, f"nexus_director_segment_{index + 1}_fallback")]
                    if name
                ),
                None,
            )
            reference_image_name = next_image
        if not reference_image_name:
            raise ValueError(f"Director segment {index + 1} is missing a reference image.")
        reference_end_image_name: str | None = None
        base_video_name: str | None = None
        parent_video = request.video or {}
        if motion.get("enabled"):
            raw_motion_name = _director_materialize_motion_video(motion, f"nexus_director_motion_{index + 1}")
            if not raw_motion_name:
                raise ValueError(f"Director segment {index + 1} has Motion Transfer enabled but no video reference.")
            base_video_name = _normalize_director_motion_reference(raw_motion_name, seconds, fps, segment_width, segment_height, frames)
            segment_request.video.update(
                {
                    "motion_transfer_enabled": True,
                    "motion_transfer_mode": "ltx_ic_union",
                    "motion_transfer_control_mode": motion_mode,
                    "motion_transfer_motion_strength": float(_number_or_none(parent_video.get("motion_transfer_motion_strength")) or 1.0),
                    "motion_transfer_target_strength": float(
                        _number_or_none(motion.get("targetStrength"))
                        if _number_or_none(motion.get("targetStrength")) is not None
                        else (1.0 if motion_mode == "camera" else (_number_or_none(parent_video.get("motion_transfer_target_strength")) if _number_or_none(parent_video.get("motion_transfer_target_strength")) is not None else 1.0))
                    ),
                    "ltx_ic_lora_strength": float(_number_or_none(parent_video.get("ltx_ic_lora_strength")) or 1.0),
                    "ltx_ic_image_bypass": bool(parent_video.get("ltx_ic_image_bypass") or False),
                    "ltx_ic_crop": parent_video.get("ltx_ic_crop") or "disabled",
                    "ltx_ic_tiled_encode": bool(parent_video.get("ltx_ic_tiled_encode") or False),
                    "ltx_ic_tile_size": int(_number_or_none(parent_video.get("ltx_ic_tile_size")) or 256),
                    "ltx_ic_tile_overlap": int(_number_or_none(parent_video.get("ltx_ic_tile_overlap")) or 64),
                }
            )
        if segment.get("transitionLoraEnabled"):
            explicit_end = _director_materialize_image_value(str(segment.get("transitionImageB64") or segment.get("transitionImageSrc") or segment.get("transitionImageFile") or ""), f"nexus_director_transition_end_{index + 1}")
            next_visual = next((item for item in visual_segments[index + 1 :] if item is not segment), None)
            reference_end_image_name = explicit_end or (_director_materialize_segment_image(next_visual, f"nexus_director_transition_next_{index + 1}") if next_visual else None)
            if not reference_end_image_name:
                raise ValueError(f"Director segment {index + 1} has Transition LoRA enabled but no end frame/reference image.")
            segment_request.video.update(
                {
                    "transition_lora_enabled": True,
                    "transition_lora": parent_video.get("transition_lora") or "Automatic",
                    "transition_lora_strength": float(_number_or_none(parent_video.get("transition_lora_strength")) or 1.0),
                    "ltx_ic_timeline_guides": bool(parent_video.get("ltx_ic_timeline_guides") or False),
                    "ltx_ic_lora_strength": float(_number_or_none(parent_video.get("ltx_ic_lora_strength")) or 1.0),
                    "ltx_ic_crop": parent_video.get("ltx_ic_crop") or "disabled",
                    "ltx_ic_tiled_encode": bool(parent_video.get("ltx_ic_tiled_encode") or False),
                    "ltx_ic_tile_size": int(_number_or_none(parent_video.get("ltx_ic_tile_size")) or 256),
                    "ltx_ic_tile_overlap": int(_number_or_none(parent_video.get("ltx_ic_tile_overlap")) or 64),
                    "start_frame_strength": float(_number_or_none(parent_video.get("start_frame_strength")) or 1.0),
                    "end_frame_strength": float(_number_or_none(parent_video.get("end_frame_strength")) or 1.0),
                }
            )
        segment_motion_mode = str(segment_request.video.get("motion_transfer_control_mode") or "pose").lower()
        ic_lora_name = assets.get("cameraman_lora") if segment_motion_mode == "camera" else assets.get("ic_lora")
        if (base_video_name or reference_end_image_name) and not ic_lora_name:
            raise ValueError("LTX Director segment render requires IC-LoRA Union Control under models/loras.")
        if job_id:
            _update_generation_job(
                job_id,
                {
                    "status": "building",
                    "progress": min(90, 10 + index * 70 // max(1, len(visual_segments))),
                    "message": f"Building LTX Director segment {index + 1}/{len(visual_segments)}",
                },
                force=True,
            )
        prompt = build_basic_ltx_img2video_workflow(
            segment_request,
            checkpoint_name,
            text_encoder_name,
            reference_image_name,
            reference_end_image_name=reference_end_image_name,
            base_video_name=base_video_name,
            ic_lora_name=ic_lora_name,
            text_projection_name=assets.get("text_projection"),
            audio_vae_name=assets.get("audio_vae"),
            video_vae_name=assets.get("video_vae") or assets.get("vae"),
            latent_upscale_name=assets.get("latent_upscale"),
            transition_lora_name=assets.get("transition_lora"),
            detailer_lora_name=assets.get("detailer_lora"),
            video_combine_node=_available_comfy_node(object_info, "VHS_VideoCombine"),
            available_nodes=available_nodes,
        )
        _apply_output_prefixes(prompt, segment_request)
        segment_started_at = datetime.now().timestamp()

        def segment_progress(update: dict[str, Any], segment_index: int = index) -> None:
            if not job_id:
                return
            if _handle_cancelled_generation_progress(job_id, update):
                return
            base = 12 + segment_index * 72 / max(1, len(visual_segments))
            span = 72 / max(1, len(visual_segments))
            progress = base + span * (float(update.get("progress") or 0) / 100.0)
            _update_generation_job(
                job_id,
                {
                    **update,
                    "progress": min(94, int(progress)),
                    "message": f"Rendering LTX Director segment {segment_index + 1}/{len(visual_segments)}",
                },
            )

        prompt_id, outputs = await comfy.run_workflow(prompt, progress_callback=segment_progress)
        last_prompt_id = prompt_id
        _raise_if_generation_cancelled(job_id)
        if not outputs:
            outputs = await _recover_outputs_from_history(prompt_id, segment_started_at)
        outputs = _cleanup_video_sidecar_images(outputs, segment_started_at)
        if not outputs:
            outputs = _cleanup_video_sidecar_images(_recent_output_files(segment_started_at - 300, limit=12), segment_started_at)
        video_path = next(
            (
                path
                for item in outputs
                for path in [_output_path_from_item(item)]
                if path and path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".avi"}
            ),
            None,
        )
        if not video_path:
            raise ValueError(f"LTX Director segment {index + 1} did not produce a video.")
        output_paths.append(video_path)
    if job_id:
        _update_generation_job(job_id, {"status": "running", "progress": 95, "message": "Joining LTX Director segments"}, force=True)
    archived_segments = _archive_director_segment_videos(output_paths, director_run_stamp)
    final_path = _concat_director_segment_videos(archived_segments, fps, width, height, director_run_stamp)
    final_path = _mux_director_audio(final_path, timeline, director_run_stamp)
    outputs = [_output_item(final_path, "video")]
    _annotate_output_metadata(outputs, request, assets)
    final_metadata = final_path.with_name(final_path.name + ".nexus.json")
    if final_metadata.exists():
        try:
            metadata = json.loads(final_metadata.read_text(encoding="utf-8"))
            metadata.setdefault("director", {})
            metadata["director"]["segment_archive"] = str((settings.output_dir / "director" / director_run_stamp / "segments").relative_to(settings.output_dir)).replace("\\", "/")
            metadata["director"]["segment_outputs"] = [
                str(path.relative_to(settings.output_dir)).replace("\\", "/")
                for path in archived_segments
                if path.exists()
            ]
            final_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return GenerateResponse(
        job_id=last_prompt_id or f"director-segments-{uuid.uuid4().hex[:10]}",
        prompt_id=last_prompt_id,
        status="completed",
        message="LTX Director segment render completed.",
        outputs=outputs,
    )


def _available_comfy_node(object_info: dict[str, Any], *names: str) -> str | None:
    available = set(object_info or {})
    for name in names:
        if name in available:
            return name
    return None


def _replace_workflow_refs(value: Any, remap: dict[str, Any]) -> Any:
    if isinstance(value, list):
        if len(value) >= 2 and str(value[0]) in remap:
            replacement = remap[str(value[0])]
            if isinstance(replacement, list):
                replacement = _replace_workflow_refs(replacement, remap)
                if isinstance(replacement, list) and len(replacement) >= 2:
                    return [replacement[0], replacement[1], *value[2:]]
                return replacement
            return replacement
        return [_replace_workflow_refs(item, remap) for item in value]
    if isinstance(value, dict):
        return {key: _replace_workflow_refs(item, remap) for key, item in value.items()}
    return value


def _bypass_missing_audio_normalization(prompt: dict[str, Any], object_info: dict[str, Any]) -> list[str]:
    if _available_comfy_node(object_info, "AudioVolumeNormalization"):
        return []
    bypassed: dict[str, Any] = {}
    titles: list[str] = []
    for node_id, node in list((prompt or {}).items()):
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type") or "") != "AudioVolumeNormalization":
            continue
        audio_ref = (node.get("inputs") or {}).get("audio")
        if not isinstance(audio_ref, list) or len(audio_ref) < 2:
            continue
        bypassed[str(node_id)] = audio_ref
        meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
        titles.append(str(meta.get("title") or node_id))
    if not bypassed:
        return []
    for node_id in list(bypassed):
        seen = {node_id}
        audio_ref = bypassed[node_id]
        while isinstance(audio_ref, list) and len(audio_ref) >= 2 and str(audio_ref[0]) in bypassed and str(audio_ref[0]) not in seen:
            seen.add(str(audio_ref[0]))
            audio_ref = bypassed[str(audio_ref[0])]
        bypassed[node_id] = audio_ref
    for node_id in bypassed:
        prompt.pop(node_id, None)
    for node in (prompt or {}).values():
        if isinstance(node, dict):
            node["inputs"] = _replace_workflow_refs(node.get("inputs") or {}, bypassed)
    return titles


def _inpaint_uses_lanpaint(request: GenerateRequest) -> bool:
    mode = request.img2img.mode.lower()
    if request.activity != "img2img" or not ("inpaint" in mode or "outpaint" in mode or "extend" in mode):
        return False
    return str(getattr(request.img2img, "inpaint_engine", "") or "").strip().lower() in {
        "lanpaint",
        "lan paint",
        "lanpaint_ksampler",
    }


def _ensure_lanpaint_custom_node(request: GenerateRequest) -> bool:
    if not _inpaint_uses_lanpaint(request):
        return False
    if (settings.custom_nodes_dir / "LanPaint").exists():
        return False
    installed, errors = install_custom_node_dependencies(
        settings,
        node_names=["https://github.com/scraed/LanPaint"],
        all_enabled=False,
    )
    if errors:
        detail = "; ".join(f"{name}: {error}" for name, error in errors.items())
        raise ValueError(f"LanPaint custom node is required for LanPaint inpaint and could not be installed: {detail}")
    if "LanPaint" not in installed and not (settings.custom_nodes_dir / "LanPaint").exists():
        raise ValueError("LanPaint custom node is required for LanPaint inpaint and could not be installed.")
    return True


def _apply_inpaint_intent_prompt(request: GenerateRequest) -> None:
    mode = request.img2img.mode.lower()
    if request.activity != "img2img" or not ("inpaint" in mode or "outpaint" in mode or "extend" in mode):
        return
    intent = str(getattr(request.img2img, "inpaint_intent", "") or "").strip().lower()
    if intent not in {"remove", "mixed"} and not getattr(request.img2img, "remove_mask_present", False):
        return
    guidance = (
        "remove the green masked objects and reconstruct the background naturally, "
        "no trace of the removed object, seamless clean inpainting"
    )
    negative = "remaining object, object silhouette, ghost object, duplicate object, mask outline, green marks"
    if guidance.lower() not in (request.prompt or "").lower():
        request.prompt = f"{request.prompt}, {guidance}" if request.prompt else guidance
    if negative.lower() not in (request.negative_prompt or "").lower():
        request.negative_prompt = f"{request.negative_prompt}, {negative}" if request.negative_prompt else negative


def _prepare_reference_image(request: GenerateRequest) -> str | None:
    value = (request.img2img.reference_image or "").strip()
    if request.activity != "img2img" or not value:
        return None
    return _prepare_reference_value(value, "nexus_reference")


def _prepare_mask_image(request: GenerateRequest) -> str | None:
    model3d_options = request.model3d if isinstance(request.model3d, dict) else {}
    model3d_texture_paint = request.preset.lower() == "model3d" and str(getattr(request, "workspace", "") or "").lower() == "texture_paint"
    model3d_mask_space = str(model3d_options.get("texture_mask_space") or "").lower()
    model3d_view_mask = str(model3d_options.get("texture_view_mask_image") or "").strip()
    model3d_uv_mask = str(model3d_options.get("texture_mask_image") or "").strip()
    model3d_value = model3d_view_mask if model3d_texture_paint and model3d_mask_space == "viewpoint" and model3d_view_mask else model3d_uv_mask
    mode = request.img2img.mode.lower()
    mask_mode = "inpaint" in mode or "outpaint" in mode or "extend" in mode
    value = (model3d_value if request.preset.lower() == "model3d" else "") or (request.img2img.mask_image or "").strip()
    if not value and ("outpaint" in mode or "extend" in mode):
        value = (getattr(request.img2img, "composite_mask_image", None) or "").strip()
    if request.activity != "img2img" or not value or (not mask_mode and not model3d_texture_paint):
        return None
    if not value.startswith("data:image/"):
        raise ValueError("Invalid inpaint mask image.")
    return _write_input_data_image(value, "nexus_texture_mask" if model3d_texture_paint else "nexus_mask", normalize=False)


def _prepare_composite_mask_image(request: GenerateRequest) -> str | None:
    value = (getattr(request.img2img, "composite_mask_image", None) or "").strip()
    mode = request.img2img.mode.lower()
    if request.activity != "img2img" or not value or not ("outpaint" in mode or "extend" in mode):
        return None
    if not value.startswith("data:image/"):
        raise ValueError("Invalid composite mask image.")
    return _write_input_data_image(value, "nexus_composite_mask", normalize=False)


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
    motion_mode = str(video.get("motion_transfer_control_mode") or "").strip().lower()
    if motion_mode == "camera" and assets.get("cameraman_lora"):
        video["ic_lora"] = assets["cameraman_lora"]
        video["motion_ic_lora"] = assets["cameraman_lora"]
    activity_label = _generation_activity_label(request)
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
        "activity": activity_label,
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


def _generation_activity_label(request: GenerateRequest) -> str:
    if request.activity != "img2img":
        return request.activity
    base_video = bool((request.img2img.base_video or "").strip())
    refs = [value for value in _reference_image_values(request) if str(value or "").strip()]
    if base_video:
        return "v2v"
    if refs and (request.preset.lower() in {"wan", "ltx"} or (request.video or {})):
        return "i2v"
    return request.activity


def _model3d_mesh_quality_report(path: Path, request: GenerateRequest) -> dict[str, Any] | None:
    if str(request.preset or "").lower() != "model3d" or path.suffix.lower() not in {".glb", ".gltf", ".obj", ".stl", ".ply"}:
        return None
    try:
        import trimesh
    except Exception:
        return {"checked": False, "reason": "trimesh_unavailable"}
    try:
        loaded = trimesh.load(path, force="scene")
        geometries = list(getattr(loaded, "geometry", {}).values()) if hasattr(loaded, "geometry") else [loaded]
        meshes = []
        for mesh in geometries:
            faces = getattr(mesh, "faces", None)
            if faces is not None and len(faces):
                meshes.append(mesh)
        if not meshes:
            return {"checked": True, "passed": False, "reason": "no_mesh_geometry"}
        mesh = trimesh.util.concatenate(meshes) if len(meshes) > 1 else meshes[0]
        face_count = int(len(mesh.faces))
        vertex_count = int(len(mesh.vertices))
        parts = mesh.split(only_watertight=False)
        component_faces = sorted((int(len(part.faces)) for part in parts), reverse=True)
        largest_faces = component_faces[0] if component_faces else face_count
        component_count = len(component_faces) or 1
        target_faces = int((request.model3d or {}).get("decimation_target") or 250000)
        min_expected_faces = max(50000, min(180000, int(target_faces * 0.18)))
        largest_ratio = (largest_faces / face_count) if face_count else 0.0
        issues: list[str] = []
        if face_count < min_expected_faces:
            issues.append("low_face_count")
        if component_count > 64 and largest_ratio < 0.8:
            issues.append("fragmented_components")
        if component_count > 160:
            issues.append("too_many_components")
        extents = [float(value) for value in getattr(mesh.bounding_box, "extents", [])]
        if len(extents) == 3 and (min(extents) <= 1e-6 or max(extents) / max(min(extents), 1e-6) > 80):
            issues.append("degenerate_bounds")
        return {
            "checked": True,
            "passed": not issues,
            "issues": issues,
            "faces": face_count,
            "vertices": vertex_count,
            "components": component_count,
            "largest_component_faces": largest_faces,
            "largest_component_ratio": round(largest_ratio, 4),
            "watertight": bool(getattr(mesh, "is_watertight", False)),
            "bounds": extents,
            "target_faces": target_faces,
            "min_expected_faces": min_expected_faces,
        }
    except Exception as exc:
        return {"checked": False, "reason": str(exc)}


def _resolve_generation_seed(request: GenerateRequest) -> int:
    if int(request.seed or -1) >= 0:
        seed = int(request.seed)
        if str(request.preset or "").lower() == "model3d":
            seed = min(seed, 0x7FFFFFFF)
            request.seed = seed
        return seed
    max_seed = 0x7FFFFFFF if str(request.preset or "").lower() == "model3d" else 2**32 - 1
    seed = random.randint(0, max_seed)
    request.seed = seed
    return seed


def _number_or_none(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


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
        quality_report = _model3d_mesh_quality_report(path, request)
        if quality_report:
            output["quality"] = quality_report
            file_metadata["model3d_quality"] = quality_report
            if quality_report.get("checked") and not quality_report.get("passed", True):
                issues = ", ".join(str(item) for item in quality_report.get("issues") or []) or str(quality_report.get("reason") or "quality_check_failed")
                output["warning"] = f"Model3D mesh QC flagged: {issues}"
                print(f"NEXUS BTA WARN Model3D mesh QC flagged {path.name}: {issues}")
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


def _safe_output_media_path(output: dict[str, Any]) -> Path | None:
    relative = str(output.get("path") or output.get("filename") or "").strip()
    if not relative:
        return None
    try:
        path = (settings.output_dir / relative).resolve()
        if not path.exists() or not path.is_relative_to(settings.output_dir.resolve()):
            return None
    except Exception:
        return None
    return path


def _input_reference_path(name: str | None) -> Path | None:
    if not name:
        return None
    try:
        path = (settings.input_dir / str(name)).resolve()
        if path.exists() and path.is_file() and path.is_relative_to(settings.input_dir.resolve()):
            return path
    except Exception:
        return None
    return None


def _director_endpoint_reference_paths(request: GenerateRequest) -> tuple[Path | None, Path | None]:
    director = getattr(request, "director", None)
    if not isinstance(director, dict):
        return None, None
    timeline = director.get("timeline_data") if isinstance(director.get("timeline_data"), dict) else {}
    visual_segments = [
        segment
        for segment in timeline.get("segments", []) or []
        if isinstance(segment, dict) and str(segment.get("sourceType") or segment.get("type") or "image").lower() in {"image", "video"}
    ]
    visual_segments.sort(key=lambda item: int(item.get("start") or 0))
    paths: list[Path] = []
    for index, segment in enumerate(visual_segments):
        image_value = str(segment.get("imageB64") or segment.get("imageSrc") or "").strip()
        image_file = str(segment.get("imageFile") or "").strip()
        if image_value.startswith("data:image/"):
            paths.append(_input_reference_path(_write_input_data_image(image_value, f"nexus_director_frame_{index + 1}")) or Path())
            continue
        if image_file:
            candidate = _input_reference_path(image_file)
            if candidate:
                paths.append(candidate)
    paths = [path for path in paths if path and path.exists()]
    if not paths:
        return None, None
    return paths[0], paths[-1] if len(paths) > 1 else paths[0]


def _replace_video_frame_with_image(frame_path: Path, image_path: Path, width: int, height: int) -> bool:
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        return False
    filter_expr = (
        f"scale={width}:{height}:force_original_aspect_ratio=decrease,"
        f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:black"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(image_path),
        "-vf",
        filter_expr,
        "-frames:v",
        "1",
        str(frame_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    return result.returncode == 0


def _video_dimensions(video_path: Path) -> tuple[int, int]:
    ffprobe = _ffprobe_binary()
    if not ffprobe:
        return 512, 512
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height",
        "-of",
        "csv=s=x:p=0",
        str(video_path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        width, height = [int(part) for part in result.stdout.strip().split("x", 1)]
        return max(1, width), max(1, height)
    except Exception:
        return 512, 512


def _lock_video_endpoint_frames(video_path: Path, start_image: Path | None, end_image: Path | None) -> bool:
    if not start_image and not end_image:
        return False
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg or not video_path.exists():
        return False
    temp_root = settings.temp_dir / f"ltx_frame_lock_{uuid.uuid4().hex[:8]}"
    frames_dir = temp_root / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    locked_video = temp_root / "locked.mp4"
    try:
        extract_command = [ffmpeg, "-y", "-i", str(video_path), str(frames_dir / "frame_%06d.png")]
        if subprocess.run(extract_command, capture_output=True, text=True).returncode != 0:
            return False
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return False
        width, height = _video_dimensions(video_path)
        changed = False
        if start_image and start_image.exists():
            changed = _replace_video_frame_with_image(frames[0], start_image, width, height) or changed
        if end_image and end_image.exists() and len(frames) > 1:
            changed = _replace_video_frame_with_image(frames[-1], end_image, width, height) or changed
        if not changed:
            return False
        fps = max(1.0, min(120.0, _ffprobe_fps(video_path)))
        encode_command = [
            ffmpeg,
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(frames_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            str(locked_video),
        ]
        if subprocess.run(encode_command, capture_output=True, text=True).returncode != 0 or not locked_video.exists():
            return False
        shutil.move(str(locked_video), str(video_path))
        return True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _make_video_seamless_forward_loop(video_path: Path, blend_frames: int | None = None) -> bool:
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg or not video_path.exists():
        return False
    temp_root = settings.temp_dir / f"ltx_loop_seam_{uuid.uuid4().hex[:8]}"
    frames_dir = temp_root / "frames"
    sequence_dir = temp_root / "sequence"
    frames_dir.mkdir(parents=True, exist_ok=True)
    sequence_dir.mkdir(parents=True, exist_ok=True)
    loop_video = temp_root / "loop.mp4"
    try:
        extract_command = [ffmpeg, "-y", "-i", str(video_path), str(frames_dir / "frame_%06d.png")]
        if subprocess.run(extract_command, capture_output=True, text=True).returncode != 0:
            return False
        frames = sorted(frames_dir.glob("frame_*.png"))
        if len(frames) < 5:
            return False
        target_count = len(frames)
        blend_count = int(blend_frames or max(8, min(30, round(target_count * 0.18))))
        blend_count = max(3, min(blend_count, target_count // 3))
        from PIL import Image

        for frame_path in frames:
            shutil.copy2(frame_path, sequence_dir / frame_path.name)

        first_frame = Image.open(frames[0]).convert("RGB")
        for offset, frame_path in enumerate(frames[-blend_count:], start=1):
            alpha = offset / blend_count
            with Image.open(frame_path).convert("RGB") as tail_frame:
                blended = Image.blend(tail_frame, first_frame, alpha)
                blended.save(sequence_dir / frame_path.name)
        fps = max(1.0, min(120.0, _ffprobe_fps(video_path)))
        encode_command = [
            ffmpeg,
            "-y",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            str(sequence_dir / "frame_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "16",
            "-an",
            str(loop_video),
        ]
        if subprocess.run(encode_command, capture_output=True, text=True).returncode != 0 or not loop_video.exists():
            return False
        shutil.move(str(loop_video), str(video_path))
        return True
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def _apply_ltx_loop_cycle_seam(outputs: list[dict[str, Any]], request: GenerateRequest) -> None:
    if request.preset.lower() != "ltx":
        return
    video_options = request.video or {}
    loop_enabled = _truthy(video_options.get("ltx_loop_cycle"))
    loop_source = str(video_options.get("ltx_loop_source") or "").strip().lower()
    if not loop_enabled or loop_source != "start_frame_as_end_frame":
        return
    if not _truthy(video_options.get("ltx_loop_post_seam_blend")):
        return
    for output in outputs:
        path = _safe_output_media_path(output)
        if not path or path.suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
            continue
        blend_frames = _number_or_none(video_options.get("ltx_loop_blend_frames"))
        if _make_video_seamless_forward_loop(path, int(blend_frames) if blend_frames is not None else None):
            output.setdefault("metadata", {})
            output["ltx_loop_seamless_forward"] = True


def _apply_ltx_reference_frame_lock(
    outputs: list[dict[str, Any]],
    request: GenerateRequest,
    reference_image_names: list[str],
) -> None:
    if request.preset.lower() != "ltx":
        return
    video_options = request.video or {}
    frame_lock_enabled = video_options.get("ltx_endpoint_frame_lock") or video_options.get("endpoint_frame_lock")
    if isinstance(frame_lock_enabled, str):
        frame_lock_enabled = frame_lock_enabled.lower() in {"true", "1", "on", "yes"}
    if not frame_lock_enabled:
        return
    start_image = _input_reference_path(reference_image_names[0]) if reference_image_names else None
    end_image = _input_reference_path(reference_image_names[1]) if len(reference_image_names) > 1 else None
    if getattr(request, "workspace", "") == "director":
        director_start, director_end = _director_endpoint_reference_paths(request)
        start_image = director_start or start_image
        end_image = director_end or end_image
    if not start_image and not end_image:
        return
    for output in outputs:
        path = _safe_output_media_path(output)
        if not path or path.suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
            continue
        if _lock_video_endpoint_frames(path, start_image, end_image):
            output.setdefault("metadata", {})
            output["ltx_frame_lock"] = True


def _normalize_ltx_start_end_motion(
    outputs: list[dict[str, Any]],
    request: GenerateRequest,
    reference_image_names: list[str],
) -> None:
    if request.preset.lower() != "ltx" or len(reference_image_names) < 2:
        return
    video_options = request.video or {}
    if _truthy(video_options.get("motion_transfer_enabled")):
        return
    if str(video_options.get("ltx_start_end_mode") or "flf_guides").lower() == "motion_scaffold":
        return
    if str(video_options.get("ltx_temporal_normalize") or "true").lower() in {"false", "0", "off", "no", "disabled"}:
        return
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg:
        return
    try:
        from PIL import Image
        import numpy as np
    except Exception:
        return
    fps = max(1.0, min(120.0, float(_number_or_none(video_options.get("fps")) or 24.0)))
    target_frames = int(round(_number_or_none(video_options.get("frames")) or 0))
    if target_frames <= 1:
        seconds = max(0.25, float(_number_or_none(video_options.get("seconds") or video_options.get("duration")) or 1.0))
        target_frames = int(round(seconds * fps)) + 1
    duration = max(0.1, target_frames / fps)

    for output in outputs:
        path = _safe_output_media_path(output)
        if not path or path.suffix.lower() not in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
            continue
        temp_root = settings.temp_dir / f"ltx_temporal_normalize_{uuid.uuid4().hex[:8]}"
        frames_dir = temp_root / "frames"
        active_dir = temp_root / "active"
        frames_dir.mkdir(parents=True, exist_ok=True)
        active_dir.mkdir(parents=True, exist_ok=True)
        normalized_video = temp_root / "normalized.mp4"
        try:
            extract = [ffmpeg, "-y", "-v", "error", "-i", str(path), str(frames_dir / "frame_%06d.png")]
            if subprocess.run(extract, capture_output=True, text=True).returncode != 0:
                continue
            frames = sorted(frames_dir.glob("frame_*.png"))
            if len(frames) < 6:
                continue
            lumas = []
            for frame in frames:
                arr = np.asarray(Image.open(frame).convert("L").resize((128, 128)), dtype=np.float32)
                lumas.append(arr)
            stack = np.stack(lumas)
            deltas = np.abs(np.diff(stack, axis=0)).mean(axis=(1, 2))
            repeat_fraction = float((deltas < 0.65).mean())
            if repeat_fraction < 0.20:
                continue
            from_start = np.abs(stack - stack[0]).mean(axis=(1, 2))
            final_delta = max(float(from_start[-1]), 1.0)
            active_end = len(frames) - 1
            for index in range(2, len(frames)):
                tail = deltas[index:]
                if from_start[index] >= final_delta * 0.88 and len(tail) and float((tail < 1.0).mean()) >= 0.45:
                    active_end = index
                    break
            if active_end < 3 or active_end >= len(frames) - 2:
                continue
            active_frames = frames[: active_end + 1]
            for index, frame in enumerate(active_frames, start=1):
                shutil.copy2(frame, active_dir / f"frame_{index:06d}.png")
            source_rate = max(1.0, (len(active_frames) - 1) / duration)
            filter_expr = (
                f"minterpolate=fps={fps:.6f}:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,"
                f"fps={fps:.6f},"
                f"tpad=stop_mode=clone:stop_duration={duration:.6f},"
                f"trim=end_frame={target_frames},"
                f"setpts=N/({fps:.6f}*TB)"
            )
            command = [
                ffmpeg,
                "-y",
                "-v",
                "error",
                "-framerate",
                f"{source_rate:.6f}",
                "-i",
                str(active_dir / "frame_%06d.png"),
                "-vf",
                filter_expr,
                "-frames:v",
                str(target_frames),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "16",
                str(normalized_video),
            ]
            if subprocess.run(command, capture_output=True, text=True).returncode != 0 or not normalized_video.exists():
                continue
            shutil.move(str(normalized_video), str(path))
            output["ltx_temporal_normalized"] = True
            output["ltx_temporal_active_frames"] = len(active_frames)
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)


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
    if class_lower in {"saveglb", "savegltf", "savemesh", "save3dmodel"}:
        return "3d"
    if "filename_prefix" not in (node.get("inputs") or {}):
        return None
    if "video" in class_lower or "gif" in class_lower:
        return "video"
    if any(token in class_lower for token in ("mesh", "glb", "gltf", "3d", "obj", "save3d")):
        return "3d"
    if "image" in class_lower or "rgba" in class_lower:
        return "image"
    return None


def _apply_output_prefixes(prompt: dict[str, Any], request: GenerateRequest) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model3d_timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    base = "_".join(
        part
        for part in (
            timestamp,
            _output_slug(request.preset, "preset"),
            _output_slug(_generation_activity_label(request), "generation"),
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
        if str(request.preset or "").lower() == "model3d":
            filename = "3DMODEL" if kind == "3d" else ("TEXTURE" if kind == "image" else kind.upper())
            inputs["filename_prefix"] = f"3D/{model3d_timestamp}/{filename}"
            continue
        current = _output_slug(Path(str(inputs.get("filename_prefix") or "")).name, "")
        suffix = f"_{current}" if current and current.lower() not in {"comfyui", "nexus_bta"} else ""
        inputs["filename_prefix"] = f"{kind}/{base}{suffix}"


def _cleanup_video_sidecar_images(outputs: list[dict[str, Any]], start_timestamp: float) -> list[dict[str, Any]]:
    if not outputs:
        return outputs
    video_root = (settings.output_dir / "video").resolve()
    try:
        output_root = settings.output_dir.resolve()
    except Exception:
        return outputs
    has_video = any(str(item.get("kind") or item.get("type") or "").lower() == "video" or str(item.get("url") or "").lower().endswith((".mp4", ".webm", ".mkv", ".mov", ".avi")) for item in outputs)
    if not has_video or not video_root.exists():
        return outputs
    image_suffixes = {".png", ".jpg", ".jpeg", ".webp"}
    deleted: set[str] = set()
    for path in video_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in image_suffixes:
            continue
        try:
            resolved = path.resolve()
            if not resolved.is_relative_to(output_root) or not resolved.is_relative_to(video_root):
                continue
            relative = path.relative_to(settings.output_dir).as_posix()
            path.unlink(missing_ok=True)
            deleted.add(relative)
        except Exception:
            continue
    if not deleted:
        return outputs
    return [item for item in outputs if str(item.get("path") or "").replace("\\", "/") not in deleted]


def _recent_output_files(start_timestamp: float, limit: int = 8) -> list[dict[str, Any]]:
    if not settings.output_dir.exists():
        return []
    model_suffixes = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz"}
    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".mkv", ".mov", ".avi", *model_suffixes}
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
        kind = "3d" if suffix in model_suffixes else ("video" if suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"} else "image")
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


def _extract_audio_to_input(video_path: Path, prefix: str = "nexus_director_audio") -> str:
    ffmpeg = _ffmpeg_binary()
    filename = f"{prefix}_{uuid.uuid4().hex[:10]}.wav"
    target = settings.input_dir / filename
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "44100",
        "-ac",
        "2",
        str(target),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0 or not target.exists() or target.stat().st_size <= 44:
        target.unlink(missing_ok=True)
        raise ValueError("Selected Director video has no readable audio track.")
    return filename


def _safe_upload_name(name: str, fallback: str = "source") -> str:
    suffix = Path(name or "").suffix.lower()
    stem = re.sub(r"[^a-zA-Z0-9._-]+", "_", Path(name or fallback).stem).strip("._-") or fallback
    return f"{stem[:48]}{suffix}"


def _copy_extras_source_to_comfy_input(source: Path, prefix: str = "extras") -> str:
    settings.input_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{prefix}_{uuid.uuid4().hex[:8]}_{_safe_upload_name(source.name, prefix)}"
    target = (settings.input_dir / filename).resolve()
    if not target.is_relative_to(settings.input_dir.resolve()):
        raise ValueError("Invalid Extras input target.")
    shutil.copy2(source, target)
    return filename


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


def _nvidia_extras_status(engine: str) -> dict[str, Any]:
    normalized = engine.strip().lower()
    info = NVIDIA_EXTRAS_ENGINES.get(normalized)
    if not info:
        raise HTTPException(status_code=404, detail=f"Unknown NVIDIA Extras engine: {engine}")
    expected_nodes = tuple(info.get("nodes") or ())
    packages = tuple(info.get("packages") or ())
    node_ready = all((settings.custom_nodes_dir / name).exists() for name in expected_nodes)
    if normalized in {"nvidia_rtx", "nvidia_pid"}:
        package_status = {name: "checked_in_comfy_runtime" for name in packages}
        dependency_ready = node_ready
    else:
        package_status = {name: importlib.util.find_spec(name) is not None for name in packages}
        dependency_ready = all(package_status.values()) if package_status else True
    result = {
        "engine": normalized,
        "label": info["label"],
        "installed": bool(node_ready and dependency_ready),
        "node_ready": node_ready,
        "dependency_ready": dependency_ready,
        "expected_nodes": list(expected_nodes),
        "packages": package_status,
        "model_required": bool(info.get("model_required")),
        "models_auto_download": normalized == "nvidia_pid",
        "notes": info.get("notes") or "",
    }
    if normalized == "nvidia_pid":
        result["pid"] = _nvidia_pid_prepare_status()
    if normalized == "nvidia_rtx":
        result["upscale_catalog"] = _ensure_nvidia_rtx_catalog_marker()
    return result


def _mark_nvidia_upscale_runtime(upscale: dict[str, Any], engine: str) -> None:
    status = _nvidia_extras_status(engine)
    upscale["runtime_engine"] = engine if status["installed"] else "standard_fallback"
    upscale["expected_nodes"] = status["expected_nodes"]
    upscale["node_ready"] = status["node_ready"]
    upscale["dependency_ready"] = status["dependency_ready"]
    upscale["model_required"] = status["model_required"]
    upscale["workflow_reference"] = status["label"]
    upscale["fallback_engine"] = "standard"
    if not status["installed"]:
        reason = "missing_custom_node" if not status["node_ready"] else "missing_dependency"
        upscale["fallback_reason"] = reason
    if engine == "nvidia_pid":
        pid_options = upscale.get("pid") if isinstance(upscale.get("pid"), dict) else {}
        pid_status = _nvidia_pid_prepare_status(str(pid_options.get("profile") or ""), pid_options)
        upscale["pid"] = pid_status["profile"]
        upscale["pid_source_ready"] = pid_status["source_ready"]
        upscale["pid_prepared"] = pid_status["prepared"]
        upscale["pid_assets"] = pid_status["assets"]
        upscale["latent_decoder_only"] = True
        upscale["media_runtime_note"] = "PiD improves latent decode/upscale. Rendered image/video sources need a latent workflow; Extras media uses restoration fallback."
        upscale["runtime_engine"] = "nvidia_pid_prepared_fallback" if pid_status["prepared"] else "standard_fallback"
        upscale["fallback_engine"] = "standard"
        upscale["fallback_reason"] = (
            "pid_requires_latent_workflow_for_direct_media_upscale"
            if pid_status["prepared"]
            else "missing_pid_source_or_assets"
        )


def _nvidia_rtx_upscale_pil(image: Any, target_size: tuple[int, int], quality: str = "HIGH") -> Any:
    import numpy as np
    import torch
    import nvvfx
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for NVIDIA RTX Video Super Resolution.")
    output_width = max(8, round(int(target_size[0]) / 8) * 8)
    output_height = max(8, round(int(target_size[1]) / 8) * 8)
    quality_mapping = {
        "LOW": nvvfx.effects.QualityLevel.LOW,
        "MEDIUM": nvvfx.effects.QualityLevel.MEDIUM,
        "HIGH": nvvfx.effects.QualityLevel.HIGH,
        "ULTRA": nvvfx.effects.QualityLevel.ULTRA,
    }
    selected_quality = quality_mapping.get(str(quality or "HIGH").upper(), nvvfx.effects.QualityLevel.HIGH)
    rgb = image.convert("RGB")
    array = np.asarray(rgb, dtype=np.float32) / 255.0
    frame = torch.from_numpy(array).cuda().permute(2, 0, 1).contiguous()
    with nvvfx.VideoSuperRes(selected_quality) as sr:
        sr.output_width = output_width
        sr.output_height = output_height
        sr.load()
        out = torch.from_dlpack(sr.run(frame).image).movedim(0, -1).detach().float().cpu().clamp(0, 1).numpy()
    return Image.fromarray((out * 255.0).round().astype("uint8"), "RGB")


def _apply_nvidia_rtx_upscale_video(source: Path, target: Path, factor: int, quality: str = "HIGH") -> bool:
    ffmpeg = _ffmpeg_binary()
    if not ffmpeg or not source.exists():
        return False
    temp_root = settings.temp_dir / "extras_nvidia_rtx" / uuid.uuid4().hex[:12]
    frames_dir = temp_root / "frames"
    upscaled_dir = temp_root / "upscaled"
    frames_dir.mkdir(parents=True, exist_ok=True)
    upscaled_dir.mkdir(parents=True, exist_ok=True)
    try:
        _run_ffmpeg([ffmpeg, "-y", "-v", "error", "-i", str(source), str(frames_dir / "frame_%06d.png")])
        frames = sorted(frames_dir.glob("frame_*.png"))
        if not frames:
            return False
        from PIL import Image

        with Image.open(frames[0]) as first:
            target_size = (int(first.width) * factor, int(first.height) * factor)
        for frame_path in frames:
            with Image.open(frame_path) as frame:
                upscaled = _nvidia_rtx_upscale_pil(frame, target_size, quality)
                upscaled.save(upscaled_dir / frame_path.name)
        fps = max(1.0, min(240.0, _ffprobe_fps(source)))
        _run_ffmpeg(
            [
                ffmpeg,
                "-y",
                "-framerate",
                f"{fps:.6f}",
                "-i",
                str(upscaled_dir / "frame_%06d.png"),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "16",
                "-an",
                str(target),
            ]
        )
        return target.exists() and target.stat().st_size > 0
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


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
    from PIL import Image, ImageEnhance, ImageFilter

    with Image.open(source) as image:
        working = image.convert("RGBA" if (plan.get("preserve_alpha") or remove_bg) else "RGB")
        mask = None
        if remove_bg:
            working, mask = _remove_bg_image_model(working, float(plan.get("remove_background", {}).get("threshold") or plan.get("threshold") or 0.45))
        else:
            upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
            upscale_enabled = bool(upscale.get("enabled") or plan.get("upscaler") or plan.get("scale"))
            if upscale_enabled:
                engine = str(upscale.get("engine") or plan.get("upscale_engine") or plan.get("upscaler") or "standard").strip().lower()
                if engine in NVIDIA_EXTRAS_ENGINES:
                    _mark_nvidia_upscale_runtime(upscale, engine)
                    plan["upscale"] = upscale
                target = _image_scale_size(working.width, working.height, plan)
                if engine == "nvidia_rtx" and upscale.get("runtime_engine") == "nvidia_rtx":
                    try:
                        working = _nvidia_rtx_upscale_pil(working, target, str(upscale.get("quality") or "HIGH"))
                        upscale["workflow_reference"] = "nvvfx.VideoSuperRes"
                        upscale.pop("fallback_reason", None)
                    except Exception as exc:
                        upscale["runtime_engine"] = "standard_fallback"
                        upscale["fallback_engine"] = "standard"
                        upscale["fallback_reason"] = f"nvidia_rtx_failed: {str(exc)[:180]}"
                        working = working.resize(target, Image.Resampling.LANCZOS)
                else:
                    working = working.resize(target, Image.Resampling.LANCZOS)
            face_restore = plan.get("face_restore")
            face_restore_enabled = bool(face_restore.get("enabled")) if isinstance(face_restore, dict) else bool(face_restore)
            detail_refine = plan.get("detail_refine") if isinstance(plan.get("detail_refine"), dict) else {}
            if detail_refine.get("enabled"):
                denoise_strength = max(0.0, min(1.0, float(_number_or_none(detail_refine.get("denoise")) or 0.18)))
                detail_strength = max(0.0, min(1.0, float(_number_or_none(detail_refine.get("detail")) or 0.30)))
                if denoise_strength > 0:
                    radius = 1 if denoise_strength < 0.35 else 2
                    working = working.filter(ImageFilter.MedianFilter(size=radius * 2 + 1))
                if detail_strength > 0:
                    working = working.filter(ImageFilter.UnsharpMask(radius=1.0 + detail_strength, percent=int(45 + detail_strength * 120), threshold=4))
                detail_refine["runtime"] = "pil_median_unsharp"
            if face_restore_enabled:
                if isinstance(face_restore, dict):
                    face_restore["runtime"] = "pil_fallback"
                working = ImageEnhance.Sharpness(working).enhance(1.08)
                working = ImageEnhance.Contrast(working.filter(ImageFilter.UnsharpMask(radius=1.1, percent=55, threshold=4))).enhance(1.02)

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


def _ffprobe_duration(source: Path) -> float:
    ffprobe = _ffprobe_binary()
    if not ffprobe:
        return 1.0
    command = [
        ffprobe,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(source),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        value = float((result.stdout or "").strip() or "0")
        return value if value > 0 else 1.0
    except Exception:
        return 1.0


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


def _extras_video_encode_settings(plan: dict[str, Any] | None = None) -> dict[str, Any]:
    plan = plan or {}
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    engine = str(upscale.get("engine") or upscale.get("runtime_engine") or "").strip().lower()
    quality = str(upscale.get("quality") or "HIGH").strip().upper()
    crf = 18
    if engine in {"flashvsr", "seedvr2", "nvidia_rtx"}:
        crf = 17 if quality == "ULTRA" else 18
    if _number_or_none(plan.get("encode_crf")) is not None:
        crf = int(max(10, min(28, float(plan.get("encode_crf")))))
    return {"crf": crf, "preset": "medium" if quality != "ULTRA" else "slow", "tune": "animation"}


def _video_encoder_args(encoder: str, output: Path, plan: dict[str, Any] | None = None) -> list[str]:
    encoder = (encoder or "mp4_h264").lower()
    if encoder == "webm_vp9":
        return ["-c:v", "libvpx-vp9", "-pix_fmt", "yuv420p", "-crf", "30", "-b:v", "0", str(output)]
    if encoder == "mov_prores_4444_alpha":
        return ["-c:v", "prores_ks", "-profile:v", "4", "-pix_fmt", "yuva444p10le", str(output)]
    if encoder == "mov_prores_422":
        return ["-c:v", "prores_ks", "-profile:v", "2", "-pix_fmt", "yuv422p10le", str(output)]
    encode = _extras_video_encode_settings(plan)
    return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", str(encode["crf"]), "-preset", str(encode["preset"]), "-tune", str(encode["tune"]), "-movflags", "+faststart", str(output)]


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


def _extras_denoise_model_ready(model_name: str) -> bool:
    text = str(model_name or "").strip()
    if not text or text.lower() in {"off", "none", "ffmpeg_hqdn3d", "nlmeans", "ffmpeg_nlmeans"}:
        return False
    candidate = Path(text)
    if candidate.exists():
        return candidate.stat().st_size > 1024 * 1024
    lowered = text.replace("\\", "/").lower()
    roots = _model_category_roots("denoise_models") + _model_category_roots("video_restore_models") + _model_category_roots("loras")
    for root in roots:
        direct = (root / text).resolve()
        if direct.exists() and direct.is_file() and direct.stat().st_size > 1024 * 1024:
            return True
        if "/" not in lowered:
            for match in root.rglob(Path(text).name):
                if match.is_file() and match.stat().st_size > 1024 * 1024:
                    return True
    return False


def _nvidia_scale_factor(upscale: dict[str, Any]) -> float:
    raw = str(upscale.get("scale") or "2x").strip().lower()
    if raw.endswith("x"):
        raw = raw[:-1]
    try:
        return max(1.0, min(4.0, float(raw)))
    except (TypeError, ValueError):
        return 2.0


def _comfy_output_items(outputs: list[dict[str, Any]], kind: str | None = None) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for output in outputs:
        relative = str(output.get("path") or "")
        if not relative:
            continue
        path = (settings.output_dir / relative).resolve()
        if path.exists() and path.is_relative_to(settings.output_dir.resolve()):
            items.append(_output_item(path, kind or output.get("kind")))
    return items


def _build_rtx_comfy_workflow(input_name: str, plan: dict[str, Any], is_video: bool) -> dict[str, Any]:
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    factor = _nvidia_scale_factor(upscale)
    quality = str(upscale.get("quality") or "ULTRA").upper()
    if quality not in {"LOW", "MEDIUM", "HIGH", "ULTRA"}:
        quality = "ULTRA"
    if is_video:
        return {
            "1": {"class_type": "LoadVideo", "inputs": {"file": input_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {"class_type": "RTXVideoSuperResolution", "inputs": {"images": ["2", 0], "resize_type": "scale by multiplier", "scale": factor, "quality": quality}},
            "4": {"class_type": "CreateVideo", "inputs": {"images": ["3", 0], "audio": ["2", 1], "fps": ["2", 2]}},
            "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": "extras/video/nvidia_rtx", "format": "auto", "codec": "auto"}},
        }
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {"class_type": "RTXVideoSuperResolution", "inputs": {"images": ["1", 0], "resize_type": "scale by multiplier", "scale": factor, "quality": quality}},
        "3": {"class_type": "SaveImage", "inputs": {"images": ["2", 0], "filename_prefix": "extras/image/nvidia_rtx"}},
    }


def _pid_source_size(source: Path, is_video: bool) -> tuple[int, int] | None:
    try:
        if is_video:
            ffprobe = _ffprobe_binary()
            if not ffprobe:
                return None
            result = subprocess.run(
                [
                    ffprobe,
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "json",
                    str(source),
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            stream = (json.loads(result.stdout).get("streams") or [{}])[0]
            width, height = int(stream.get("width") or 0), int(stream.get("height") or 0)
        else:
            from PIL import Image

            with Image.open(source) as image:
                width, height = image.size
        if width <= 0 or height <= 0:
            return None
        return width, height
    except Exception:
        return None


def _pid_aligned_size_for(width: int, height: int, align: int = 64) -> tuple[int, int] | None:
    try:
        if width <= 0 or height <= 0:
            return None
        align = 64
        width_candidates = {
            max(align, (width // align) * align),
            max(align, ((width + align - 1) // align) * align),
        }
        height_candidates = {
            max(align, (height // align) * align),
            max(align, ((height + align - 1) // align) * align),
        }
        source_aspect = width / height
        aligned_w, aligned_h = min(
            ((candidate_w, candidate_h) for candidate_w in width_candidates for candidate_h in height_candidates),
            key=lambda item: (abs((item[0] / item[1]) - source_aspect), abs((item[0] * item[1]) - (width * height))),
        )
        return aligned_w, aligned_h
    except Exception:
        return None


def _pid_media_sizes(source_path: Path | None, plan: dict[str, Any], is_video: bool, pid: dict[str, Any]) -> tuple[int, int, int, int, int]:
    source_size = _pid_source_size(source_path, is_video) if source_path else None
    source_w, source_h = source_size or (512, 512)
    ui_factor = _nvidia_scale_factor(plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {})
    target_w = max(64, int(round(source_w * ui_factor)))
    target_h = max(64, int(round(source_h * ui_factor)))

    checkpoint = str(pid.get("checkpoint") or "2k").lower()
    requested_scale = int(_number_or_none(pid.get("scale")) or 0)
    pid_scale = 4 if checkpoint in {"2k", "2kto4k"} else max(1, requested_scale or 4)
    base_w = max(64, int(round(target_w / max(1, pid_scale))))
    base_h = max(64, int(round(target_h / max(1, pid_scale))))
    max_base_long = 1024 if checkpoint == "2kto4k" else 512
    current_long = max(base_w, base_h)
    if current_long > max_base_long:
        ratio = max_base_long / current_long
        base_w = max(64, int(round(base_w * ratio)))
        base_h = max(64, int(round(base_h * ratio)))
    aligned = _pid_aligned_size_for(base_w, base_h) or (base_w, base_h)
    return int(aligned[0]), int(aligned[1]), target_w, target_h, pid_scale


def _build_pid_comfy_workflow(input_name: str, plan: dict[str, Any], is_video: bool, source_path: Path | None = None) -> dict[str, Any]:
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    pid = upscale.get("pid") if isinstance(upscale.get("pid"), dict) else {}
    backbone = str(pid.get("backbone") or "zimage")
    checkpoint = str(pid.get("checkpoint") or "2k")
    steps = int(_number_or_none(pid.get("steps")) or 4)
    cfg = float(_number_or_none(pid.get("cfg")) or 1.0)
    caption = str(plan.get("prompt") or pid.get("caption") or "high quality restored image, clean detail, low noise")
    sequential = str(pid.get("sequential_offload") or "sequential_blocks_aggressive")
    base_w, base_h, target_w, target_h, scale = _pid_media_sizes(source_path, plan, is_video, pid)
    if is_video:
        return {
            "1": {"class_type": "LoadVideo", "inputs": {"file": input_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {"class_type": "ImageScale", "inputs": {"image": ["2", 0], "upscale_method": "lanczos", "width": base_w, "height": base_h, "crop": "disabled"}},
            "4": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
            "5": {"class_type": "VAEEncode", "inputs": {"pixels": ["3", 0], "vae": ["4", 0]}},
            "6": {
                "class_type": "PiDPrepare",
                "inputs": {
                    "latent": ["5", 0],
                    "caption": caption,
                    "backbone": backbone,
                    "pid_ckpt_type": checkpoint,
                    "scale": scale,
                    "sigma": 0.0,
                    "auto_download": False,
                    "cleanup_after_prepare": True,
                },
            },
            "7": {
                "class_type": "PiDSample",
                "inputs": {
                    "prepared": ["6", 0],
                    "pid_steps": steps,
                    "cfg_scale": cfg,
                    "seed": int(time.time()) % (2**31 - 1),
                    "aggressive_cleanup": True,
                    "sequential_offload": sequential,
                },
            },
            "8": {"class_type": "PiDFinalize", "inputs": {"sampled": ["7", 0]}},
            "9": {"class_type": "ImageScale", "inputs": {"image": ["8", 0], "upscale_method": "lanczos", "width": target_w, "height": target_h, "crop": "disabled"}},
            "10": {"class_type": "CreateVideo", "inputs": {"images": ["9", 0], "audio": ["2", 1], "fps": ["2", 2]}},
            "11": {"class_type": "SaveVideo", "inputs": {"video": ["10", 0], "filename_prefix": "extras/video/nvidia_pid", "format": "auto", "codec": "auto"}},
        }
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": input_name}},
        "2": {"class_type": "ImageScale", "inputs": {"image": ["1", 0], "upscale_method": "lanczos", "width": base_w, "height": base_h, "crop": "disabled"}},
        "3": {"class_type": "VAELoader", "inputs": {"vae_name": "ae.safetensors"}},
        "4": {"class_type": "VAEEncode", "inputs": {"pixels": ["2", 0], "vae": ["3", 0]}},
        "5": {
            "class_type": "PiDPrepare",
            "inputs": {
                "latent": ["4", 0],
                "caption": caption,
                "backbone": backbone,
                "pid_ckpt_type": checkpoint,
                "scale": scale,
                "sigma": 0.0,
                "auto_download": False,
                "cleanup_after_prepare": True,
            },
        },
        "6": {
            "class_type": "PiDSample",
            "inputs": {
                "prepared": ["5", 0],
                "pid_steps": steps,
                "cfg_scale": cfg,
                "seed": int(time.time()) % (2**31 - 1),
                "aggressive_cleanup": True,
                "sequential_offload": sequential,
            },
        },
        "7": {"class_type": "PiDFinalize", "inputs": {"sampled": ["6", 0]}},
        "8": {"class_type": "ImageScale", "inputs": {"image": ["7", 0], "upscale_method": "lanczos", "width": target_w, "height": target_h, "crop": "disabled"}},
        "9": {"class_type": "SaveImage", "inputs": {"images": ["8", 0], "filename_prefix": "extras/image/nvidia_pid"}},
    }


def _build_video_restore_comfy_workflow(input_name: str, plan: dict[str, Any], engine: str, source_path: Path | None = None) -> dict[str, Any]:
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    factor = 4 if str(upscale.get("scale") or "2x").startswith("4") else 2
    source_size = _pid_source_size(source_path, True) if source_path else None
    source_w, source_h = source_size or (928, 480)
    target_short = max(256, int(round(min(source_w, source_h) * factor)))
    if engine == "flashvsr":
        if str(upscale.get("workflow") or "").lower() in {"wavespeed", "reference", "api"}:
            target_resolution = "4K" if factor >= 4 else "1080p"
            upscale["workflow_reference"] = "WavespeedFlashVSRNode"
            upscale["quality_profile"] = "wavespeed_reference"
            return {
                "1": {"class_type": "LoadVideo", "inputs": {"file": input_name}},
                "2": {"class_type": "WavespeedFlashVSRNode", "inputs": {"video": ["1", 0], "target_resolution": target_resolution}},
                "3": {"class_type": "SaveVideo", "inputs": {"video": ["2", 0], "filename_prefix": "extras/video/flashvsr", "format": "auto", "codec": "auto"}},
            }
        scale = 4 if factor >= 4 else 2
        quality_text = " ".join(str(upscale.get(key) or "") for key in ("quality", "model", "restore_model")).lower()
        quality_mode = any(token in quality_text for token in ("ultra", "full", "best", "quality", "high"))
        low_vram = any(token in quality_text for token in ("low", "tiny", "fast", "lite"))
        if not quality_mode and not low_vram:
            low_vram = True
        model_version = "Tiny Long (Low VRAM)" if low_vram else "Full (Best Quality)"
        speed_optimization = 2.0
        quality_boost = 2.0 if low_vram else 2.2
        tile_hint = int(_number_or_none(upscale.get("tile")) or (384 if low_vram else 256))
        tile_size = max(128, min(1024 if low_vram else 256, tile_hint))
        tile_overlap = 48 if low_vram and tile_size >= 384 else 24
        upscale["quality_profile"] = "low_vram" if low_vram else "full_best_quality"
        upscale["flashvsr_model_version"] = model_version
        return {
            "1": {"class_type": "LoadVideo", "inputs": {"file": input_name}},
            "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
            "3": {
                "class_type": "AILab_FlashVSR_Advanced",
                "inputs": {
                    "frames": ["2", 0],
                    "audio": ["2", 1],
                    "model_version": model_version,
                    "scale": scale,
                    "enable_tiling": True,
                    "tile_size": tile_size,
                    "tile_overlap": tile_overlap,
                    "speed_optimization": speed_optimization,
                    "quality_boost": quality_boost,
                    "stability_level": 11,
                    "color_fix": True,
                    "vae_tiling": True,
                    "unload_model": False,
                    "sageattention": "enable",
                    "device": "auto",
                    "precision": "bf16",
                    "seed": int(time.time()) % (2**31 - 1),
                },
            },
            "4": {"class_type": "CreateVideo", "inputs": {"images": ["3", 0], "audio": ["3", 1], "fps": ["2", 2]}},
            "5": {"class_type": "SaveVideo", "inputs": {"video": ["4", 0], "filename_prefix": "extras/video/flashvsr", "format": "auto", "codec": "auto"}},
        }
    seed_attention = "sageattn_2" if importlib.util.find_spec("sageattention") and importlib.util.find_spec("triton") else "sdpa"
    seed_batch = 5
    return {
        "1": {"class_type": "LoadVideo", "inputs": {"file": input_name}},
        "2": {"class_type": "GetVideoComponents", "inputs": {"video": ["1", 0]}},
        "3": {"class_type": "SeedVR2LoadDiTModel", "inputs": {"model": "seedvr2_ema_3b_fp8_e4m3fn.safetensors", "device": "cuda:0", "offload_device": "cpu", "cache_model": False, "attention_mode": seed_attention}},
        "4": {"class_type": "SeedVR2LoadVAEModel", "inputs": {"model": "ema_vae_fp16.safetensors", "device": "cuda:0", "encode_tiled": True, "encode_tile_size": 512, "encode_tile_overlap": 64, "decode_tiled": True, "decode_tile_size": 512, "decode_tile_overlap": 64, "offload_device": "cpu", "cache_model": False}},
        "5": {"class_type": "SeedVR2VideoUpscaler", "inputs": {"image": ["2", 0], "dit": ["3", 0], "vae": ["4", 0], "seed": int(time.time()) % (2**31 - 1), "resolution": target_short, "max_resolution": max(source_w, source_h) * factor, "batch_size": seed_batch, "uniform_batch_size": False, "color_correction": "lab", "temporal_overlap": 0, "offload_device": "cpu", "enable_debug": False}},
        "6": {"class_type": "CreateVideo", "inputs": {"images": ["5", 0], "audio": ["2", 1], "fps": ["2", 2]}},
        "7": {"class_type": "SaveVideo", "inputs": {"video": ["6", 0], "filename_prefix": "extras/video/seedvr2", "format": "auto", "codec": "auto"}},
    }


def _extras_comfy_timeout_seconds(plan: dict[str, Any], engine: str) -> int:
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    raw = upscale.get("timeout_seconds") or plan.get("timeout_seconds") or os.environ.get("NEXUS_EXTRAS_UPSCALE_TIMEOUT")
    try:
        if raw is not None:
            return int(max(60, min(3600, float(raw))))
    except (TypeError, ValueError):
        pass
    return 600 if engine in {"flashvsr", "seedvr2"} else 900


async def _process_video_restore_with_comfy(source_files: list[Path], plan: dict[str, Any], engine: str) -> list[dict[str, Any]]:
    if engine not in {"flashvsr", "seedvr2"} or not source_files:
        return []
    await comfy.ensure_running()
    object_info = await comfy.object_info()
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    flash_reference = engine == "flashvsr" and str(upscale.get("workflow") or "").lower() in {"wavespeed", "reference", "api"}
    required = ["LoadVideo", "SaveVideo"]
    if flash_reference:
        required.append("WavespeedFlashVSRNode")
    else:
        required.extend(["GetVideoComponents", "CreateVideo"])
        required.extend(["AILab_FlashVSR_Advanced"] if engine == "flashvsr" else ["SeedVR2LoadDiTModel", "SeedVR2LoadVAEModel", "SeedVR2VideoUpscaler"])
    missing = [node for node in required if not _available_comfy_node(object_info, node)]
    if missing:
        upscale["runtime_engine"] = "missing_nodes"
        upscale["fallback_reason"] = "missing_comfy_nodes:" + ",".join(sorted(set(missing)))
        raise RuntimeError(upscale["fallback_reason"])
    source_name = _copy_extras_source_to_comfy_input(source_files[0], engine)
    workflow = _build_video_restore_comfy_workflow(source_name, plan, engine, source_files[0])
    timeout_seconds = _extras_comfy_timeout_seconds(plan, engine)
    try:
        prompt_id, comfy_outputs = await comfy.run_workflow(workflow, timeout_seconds=timeout_seconds)
    except (TimeoutError, asyncio.TimeoutError):
        try:
            await comfy.clear_queue()
            await comfy.free_memory(unload_models=True, free_memory=True)
        except Exception:
            pass
        upscale["runtime_engine"] = "timeout"
        upscale["timeout_seconds"] = timeout_seconds
        raise TimeoutError(f"{engine} upscale timed out after {timeout_seconds}s.")
    outputs = _comfy_output_items(comfy_outputs, "video")
    if outputs:
        upscale["runtime_engine"] = engine
        upscale["comfy_prompt_id"] = prompt_id
        upscale["timeout_seconds"] = timeout_seconds
        upscale["workflow_reference"] = upscale.get("workflow_reference") or ("AILab_FlashVSR_Advanced" if engine == "flashvsr" else "SeedVR2VideoUpscaler")
    return outputs


async def _process_nvidia_extras_with_comfy(source_files: list[Path], plan: dict[str, Any], is_video: bool) -> list[dict[str, Any]]:
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    engine = str(upscale.get("engine") or "").strip().lower()
    if engine not in {"nvidia_rtx", "nvidia_pid"} or not source_files:
        return []
    if is_video and engine == "nvidia_pid":
        upscale["runtime_engine"] = "disabled"
        upscale["fallback_reason"] = "nvidia_pid_image_only"
        raise ValueError("NVIDIA PiD is image-only in Extras for now.")
    await comfy.ensure_running()
    object_info = await comfy.object_info()
    required = ["RTXVideoSuperResolution"] if engine == "nvidia_rtx" else ["PiDPrepare", "PiDSample", "PiDFinalize", "VAELoader", "VAEEncode", "ImageScale"]
    missing = [node for node in required if not _available_comfy_node(object_info, node)]
    if is_video:
        missing.extend(node for node in ("LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo") if not _available_comfy_node(object_info, node))
    else:
        missing.extend(node for node in ("LoadImage", "SaveImage") if not _available_comfy_node(object_info, node))
    if missing:
        upscale["runtime_engine"] = "standard_fallback"
        upscale["fallback_reason"] = "missing_comfy_nodes:" + ",".join(sorted(set(missing)))
        return []
    if engine == "nvidia_pid":
        pid_options = upscale.get("pid") if isinstance(upscale.get("pid"), dict) else {}
        pid_status = _nvidia_pid_prepare_status(str(pid_options.get("profile") or ""), pid_options)
        upscale["pid"] = pid_status.get("profile")
        upscale["pid_assets"] = pid_status.get("assets")
        if not pid_status.get("prepared"):
            upscale["runtime_engine"] = "standard_fallback"
            upscale["fallback_reason"] = "missing_pid_source_or_assets"
            return []
    source_name = _copy_extras_source_to_comfy_input(source_files[0], engine)
    workflow = _build_rtx_comfy_workflow(source_name, plan, is_video) if engine == "nvidia_rtx" else _build_pid_comfy_workflow(source_name, plan, is_video, source_files[0])
    prompt_id, comfy_outputs = await comfy.run_workflow(workflow, timeout_seconds=5400)
    outputs = _comfy_output_items(comfy_outputs, "video" if is_video else "image")
    if outputs:
        upscale["runtime_engine"] = engine
        upscale["comfy_prompt_id"] = prompt_id
        upscale["workflow_reference"] = "RTXVideoSuperResolution" if engine == "nvidia_rtx" else "PiDPrepare/PiDSample/PiDFinalize"
        if engine == "nvidia_pid":
            upscale["latent_decode"] = True
            upscale["staged_decode"] = True
    return outputs


def _extras_video_denoise_filters(denoise: dict[str, Any], upscale_engine: str = "") -> list[str]:
    if not denoise.get("enabled"):
        return []
    model_name = str(denoise.get("model") or "ffmpeg_hqdn3d")
    model_lower = model_name.lower()
    strength = max(0.0, min(1.0, float(_number_or_none(denoise.get("strength")) or 0.2)))
    model_ready = _extras_denoise_model_ready(model_name)
    wants_quality = model_ready or any(token in model_lower for token in ("fastdvd", "dvdnet", "swinir", "vrt", "basicvsr", "seedvr", "restore", "detailer", "lora"))
    if "ltx" in model_lower and ("detailer" in model_lower or "lora" in model_lower):
        denoise["model_runtime"] = "ltx_detailer_lora_detected_latent_only"
        denoise["runtime_note"] = "LTX detailer is a generation/latent LoRA; post-render Extras uses temporal restoration fallback."
    elif model_ready:
        denoise["model_runtime"] = "model_guided_ffmpeg_restoration"
    else:
        denoise["model_runtime"] = "ffmpeg_restoration"
    filters: list[str] = []
    if wants_quality:
        temporal_a = max(0.006, min(0.05, 0.008 + strength * 0.055))
        temporal_b = max(0.018, min(0.16, 0.025 + strength * 0.14))
        filters.append(
            "atadenoise="
            f"0a={temporal_a:.4f}:0b={temporal_b:.4f}:"
            f"1a={temporal_a * 0.72:.4f}:1b={temporal_b * 0.72:.4f}:"
            f"2a={temporal_a * 0.72:.4f}:2b={temporal_b * 0.72:.4f}:s={9 if strength < 0.45 else 13}"
        )
        filters.append(f"nlmeans=s={max(1.2, 2.0 + strength * 7.0):.3f}:p=7:r={13 if strength < 0.5 else 15}")
        filters.append(
            "deband="
            f"1thr={max(0.008, 0.012 + strength * 0.035):.4f}:"
            f"2thr={max(0.006, 0.010 + strength * 0.025):.4f}:"
            f"3thr={max(0.006, 0.010 + strength * 0.025):.4f}:"
            f"4thr={max(0.006, 0.010 + strength * 0.025):.4f}"
        )
    else:
        spatial = max(0.2, 5.5 * strength)
        temporal = max(0.6, 24.0 * strength)
        filters.append(f"hqdn3d={spatial:.3f}:{spatial:.3f}:{temporal:.3f}:{temporal:.3f}")
    if upscale_engine in {"nvidia_pid", "nvidia_rtx"} and strength > 0:
        wave = max(1.0, min(7.0, 1.6 + strength * 5.5))
        filters.append(f"vaguedenoiser=threshold={wave:.3f}:method=garrote:nsteps=6:percent={max(40, min(88, 55 + strength * 30)):.1f}")
        denoise["nvidia_cleanup"] = True
    denoise["runtime"] = ",".join(filters)
    return filters


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
    temp_roots: list[Path] = []
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
        interpolate["runtime"] = "rife_frame_interpolation"
        interpolate["source_fps_detected"] = detected_fps
        interpolate["target_fps_applied"] = active_fps
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
    post_filters: list[str] = []
    upscale = plan.get("upscale") or {}
    upscale_engine_for_denoise = str(upscale.get("engine") or upscale.get("mode") or "").strip().lower()
    denoise = plan.get("denoise") or {}
    denoise_filters = _extras_video_denoise_filters(denoise, upscale_engine_for_denoise)
    filters.extend(denoise_filters)
    if upscale.get("enabled"):
        engine = str(upscale.get("engine") or upscale.get("mode") or "standard").strip().lower()
        if engine not in {"standard", "flashvsr", "seedvr2", "ltx_detailer", "nvidia_rtx", "nvidia_pid"}:
            engine = "standard"
        factor = 4 if str(upscale.get("scale") or "2x").startswith("4") else 2
        scale_filter_required = True
        if engine in {"flashvsr", "seedvr2", "ltx_detailer"}:
            target = _ltx_hf_lora_installed_path(engine)
            model_ready = target.exists() and target.stat().st_size > 1024 * 1024
            custom_node_names = EXTRAS_VIDEO_RESTORE_NODES[engine]
            node_ready = any((settings.custom_nodes_dir / name).exists() for name in custom_node_names)
            upscale["runtime_engine"] = engine if model_ready and node_ready else "standard_fallback"
            upscale["expected_nodes"] = list(custom_node_names)
            upscale["node_ready"] = node_ready
            if engine == "ltx_detailer":
                upscale["detailer_lora"] = target.name if model_ready else "ltx-2-19b-ic-lora-detailer.safetensors"
                upscale["controlnet"] = "Off"
                upscale["image_bypass"] = True
                upscale["workflow_reference"] = "LTX-2.3 Image + Audio + Video (IC-LoRA) to Video"
            if not model_ready:
                upscale["fallback_reason"] = "missing_model"
            elif not node_ready:
                upscale["fallback_reason"] = "missing_custom_node"
        elif engine in NVIDIA_EXTRAS_ENGINES:
            _mark_nvidia_upscale_runtime(upscale, engine)
            if engine == "nvidia_rtx" and upscale.get("runtime_engine") == "nvidia_rtx":
                rtx_root = settings.temp_dir / "extras_nvidia_rtx_source" / uuid.uuid4().hex[:12]
                rtx_root.mkdir(parents=True, exist_ok=True)
                temp_roots.append(rtx_root)
                normalized_source = rtx_root / "source.mp4"
                rtx_source = rtx_root / "rtx.mp4"
                try:
                    _run_ffmpeg([ffmpeg, "-y", *input_args, "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "16", "-an", str(normalized_source)])
                    if _apply_nvidia_rtx_upscale_video(normalized_source, rtx_source, factor, str(upscale.get("quality") or "HIGH")):
                        input_args = ["-i", str(rtx_source)]
                        active_fps = _ffprobe_fps(rtx_source)
                        scale_filter_required = False
                        upscale["runtime_engine"] = "nvidia_rtx"
                        upscale["workflow_reference"] = "nvvfx.VideoSuperRes"
                        upscale.pop("fallback_reason", None)
                    else:
                        upscale["runtime_engine"] = "standard_fallback"
                        upscale["fallback_reason"] = "nvidia_rtx_failed"
                except Exception as exc:
                    upscale["runtime_engine"] = "standard_fallback"
                    upscale["fallback_engine"] = "standard"
                    upscale["fallback_reason"] = f"nvidia_rtx_failed: {str(exc)[:180]}"
        else:
            upscale["runtime_engine"] = "standard"
        if scale_filter_required:
            filters.append(f"scale=iw*{factor}:ih*{factor}:flags=lanczos")
    detail_refine = plan.get("detail_refine") if isinstance(plan.get("detail_refine"), dict) else {}
    if detail_refine.get("enabled"):
        strength = max(0.0, min(1.0, float(_number_or_none(detail_refine.get("strength")) or 0.30)))
        amount = max(0.05, min(1.0, 0.18 + strength * 0.55))
        post_filters.append(f"unsharp=5:5:{amount:.3f}:3:3:{amount * 0.45:.3f}")
        detail_refine["runtime"] = "ffmpeg_unsharp"
    face_restore = plan.get("face_restore") if isinstance(plan.get("face_restore"), dict) else {}
    if face_restore.get("enabled"):
        target = _ltx_hf_lora_installed_path("face_restore")
        has_model = target.exists() and target.stat().st_size > 1024 * 1024
        face_restore["runtime"] = "ffmpeg_fallback" if has_model else "ffmpeg_fallback_missing_model"
        post_filters.append("unsharp=5:5:0.35:3:3:0.15")
    filters.extend(post_filters)
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
        for temp_root in temp_roots:
            shutil.rmtree(temp_root, ignore_errors=True)
        return outputs

    suffix = ".webm" if encoder == "webm_vp9" else ".mov" if encoder.startswith("mov_") else ".mp4"
    output = _extras_output("video", suffix, "remove_bg_video" if remove_bg else "video")
    encode_settings = _extras_video_encode_settings(plan)
    plan["encode_crf"] = encode_settings["crf"]
    plan["encode_preset"] = encode_settings["preset"]
    plan["encode_tune"] = encode_settings["tune"]
    _run_ffmpeg([*command, *_video_encoder_args(encoder, output, plan)])
    outputs.append(_output_item(output, "video"))
    for temp_root in temp_roots:
        shutil.rmtree(temp_root, ignore_errors=True)
    return outputs


def _extras_video_output_paths(outputs: list[dict[str, Any]]) -> list[Path]:
    paths: list[Path] = []
    output_root = settings.output_dir.resolve()
    for output in outputs:
        relative = str(output.get("path") or "").strip()
        if not relative:
            continue
        path = (settings.output_dir / relative).resolve()
        if path.exists() and path.is_relative_to(output_root) and path.suffix.lower() in {".mp4", ".mov", ".webm", ".mkv", ".avi"}:
            paths.append(path)
    return paths


def _extras_video_needs_post_refine(plan: dict[str, Any]) -> bool:
    for key in ("interpolate", "denoise", "detail_refine", "face_restore"):
        value = plan.get(key)
        if isinstance(value, dict) and value.get("enabled"):
            return True
    return bool(plan.get("preserve_alpha"))


def _normalize_extras_video_timing(source_files: list[Path], plan: dict[str, Any]) -> None:
    video_exts = {".mp4", ".mov", ".webm", ".mkv", ".avi"}
    source = next((path for path in source_files if path.suffix.lower() in video_exts and path.exists()), None)
    if not source:
        return
    detected_fps = _ffprobe_fps(source)
    if detected_fps <= 0:
        return
    requested_fps = _number_or_none(plan.get("source_fps"))
    if requested_fps and abs(float(requested_fps) - detected_fps) > 0.05:
        plan["source_fps_requested"] = float(requested_fps)
    plan["source_fps"] = detected_fps
    interpolate = plan.get("interpolate") if isinstance(plan.get("interpolate"), dict) else {}
    requested_interpolate_fps = _number_or_none(interpolate.get("source_fps"))
    if requested_interpolate_fps and abs(float(requested_interpolate_fps) - detected_fps) > 0.05:
        interpolate["source_fps_requested"] = float(requested_interpolate_fps)
    interpolate["source_fps"] = detected_fps
    plan["interpolate"] = interpolate


async def _postprocess_restored_video_outputs(outputs: list[dict[str, Any]], plan: dict[str, Any]) -> list[dict[str, Any]]:
    source_paths = _extras_video_output_paths(outputs)
    if not source_paths:
        return outputs
    refine_plan = json.loads(json.dumps(plan))
    refine_upscale = refine_plan.get("upscale") if isinstance(refine_plan.get("upscale"), dict) else {}
    refine_upscale["enabled"] = False
    refine_plan["upscale"] = refine_upscale
    refined = await asyncio.to_thread(_process_extras_video, source_paths, refine_plan, False)
    if not refined:
        return outputs
    for key in ("interpolate", "denoise", "detail_refine", "face_restore"):
        if isinstance(refine_plan.get(key), dict):
            target = plan.get(key) if isinstance(plan.get(key), dict) else {}
            target.update(refine_plan[key])
            plan[key] = target
    upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
    upscale["post_refine_applied"] = True
    upscale["post_refine_source"] = "comfy_restore_output"
    plan["upscale"] = upscale
    output_root = settings.output_dir.resolve()
    for path in source_paths:
        if path.is_relative_to(output_root) and path.parent.name == "video" and path.name.startswith(("flashvsr_", "seedvr2_")):
            try:
                path.unlink(missing_ok=True)
                path.with_suffix(path.suffix + ".nexus.json").unlink(missing_ok=True)
            except Exception:
                pass
    return refined


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
            _normalize_extras_video_timing(source_files, plan)
        outputs: list[dict[str, Any]] = []
        upscale = plan.get("upscale") if isinstance(plan.get("upscale"), dict) else {}
        nvidia_engine = str(upscale.get("engine") or "").strip().lower()
        if is_video and nvidia_engine == "nvidia_pid":
            raise ValueError("NVIDIA PiD is disabled for video Extras; use RTX, FlashVSR, SeedVR2, or Standard for video upscale.")
        video_restore_engine = nvidia_engine if nvidia_engine in {"flashvsr", "seedvr2"} else ""
        if not remove_bg and nvidia_engine == "nvidia_pid":
            _update_extras_job(job_id, {"progress": 22, "message": f"Running {nvidia_engine} Extras through ComfyUI."})
            try:
                outputs = await _process_nvidia_extras_with_comfy(source_files, plan, is_video)
            except Exception as exc:
                upscale["runtime_engine"] = "standard_fallback"
                upscale["fallback_reason"] = f"comfy_{nvidia_engine}_failed: {str(exc)[:240]}"
                plan["upscale"] = upscale
                outputs = []
        if not outputs and not remove_bg and is_video and video_restore_engine:
            _update_extras_job(job_id, {"progress": 22, "message": f"Running {video_restore_engine} Extras through ComfyUI."})
            try:
                outputs = await _process_video_restore_with_comfy(source_files, plan, video_restore_engine)
            except Exception as exc:
                upscale["runtime_engine"] = "failed"
                upscale["fallback_reason"] = f"comfy_{video_restore_engine}_failed: {str(exc)[:240]}"
                plan["upscale"] = upscale
                raise
        if outputs:
            if is_video and video_restore_engine:
                _update_extras_job(job_id, {"progress": 72, "message": "Applying Extras refine to restored video."})
                outputs = await _postprocess_restored_video_outputs(outputs, plan)
        elif is_video:
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
    for pattern in ("nexus_reference_*", "nexus_base_video_*", "nexus_mask_*", "nexus_controlnet_*", "nexus_director_audio_*", "nexus_director_frame_*"):
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
    next_enable_sage = bool(getattr(options, "enable_sage_attention", True))
    next_enable_flash = bool(getattr(options, "enable_flash_attention", False))
    next_profile_version = max(2, int(getattr(options, "acceleration_profile_version", 2) or 2))
    changed = (
        _canonical_vram_policy(settings.runtime.vram_policy) != next_vram
        or _canonical_gpu_memory_gb(settings.runtime.gpu_memory_gb) != next_gpu_memory_gb
        or _canonical_attention_backend(settings.runtime.attention_backend) != next_attention
        or _canonical_precision(settings.runtime.precision) != next_precision
        or bool(settings.runtime.disable_xformers) != next_disable_xformers
        or bool(settings.runtime.enable_sage_attention) != next_enable_sage
        or bool(settings.runtime.enable_flash_attention) != next_enable_flash
        or int(getattr(settings.runtime, "acceleration_profile_version", 0) or 0) != next_profile_version
    )
    settings.runtime.vram_policy = next_vram
    settings.runtime.gpu_memory_gb = next_gpu_memory_gb
    settings.runtime.attention_backend = next_attention
    settings.runtime.precision = next_precision
    settings.runtime.disable_xformers = next_disable_xformers
    settings.runtime.enable_sage_attention = next_enable_sage or next_attention == "sage"
    settings.runtime.enable_flash_attention = next_enable_flash or next_attention == "flash"
    settings.runtime.acceleration_profile_version = next_profile_version
    if changed:
        save_settings(settings)
    return changed


def _prepare_runtime_for_generation(request: GenerateRequest) -> None:
    if request.runtime.attention_backend in {"", None}:
        request.runtime.attention_backend = "auto"


def _generation_timeout_seconds(request: GenerateRequest) -> int:
    if request.preset.lower() != "model3d":
        return 3600
    options = request.model3d if isinstance(request.model3d, dict) else {}
    value = options.get("timeout_seconds") or options.get("max_runtime_seconds")
    try:
        parsed = int(float(value)) if value not in (None, "") else 600
    except (TypeError, ValueError):
        parsed = 600
    return max(180, min(parsed, 1800))


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


def _nvidia_smi_memory_snapshot() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu,pstate",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=6,
            check=False,
        )
    except Exception as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {str(exc)[:160]}"}

    if completed.returncode != 0:
        return {"available": False, "error": (completed.stderr or completed.stdout or "").strip()[:240]}

    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        return {"available": False, "error": "nvidia-smi returned no GPU rows."}

    parts = [part.strip() for part in lines[0].split(",")]
    if len(parts) < 7:
        return {"available": False, "error": f"Unexpected nvidia-smi output: {lines[0][:160]}"}

    def as_int(value: str) -> int | None:
        try:
            return int(float(value))
        except Exception:
            return None

    return {
        "available": True,
        "name": parts[0],
        "total_mb": as_int(parts[1]),
        "used_mb": as_int(parts[2]),
        "free_mb": as_int(parts[3]),
        "utilization_gpu_percent": as_int(parts[4]),
        "temperature_c": as_int(parts[5]),
        "pstate": parts[6],
        "note": "Per-process VRAM may be unavailable on Windows WDDM; this is the GPU total snapshot.",
    }


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
        "nvidia_smi": _nvidia_smi_memory_snapshot(),
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
            if module_name == "xformers":
                import xformers.ops  # noqa: F401

                if importlib.util.find_spec("xformers._C") is None:
                    result["available"] = False
                    result["error"] = "xFormers CUDA/C++ extension is not available."
            elif module_name == "sageattention":
                if importlib.util.find_spec("triton") is None:
                    result["available"] = False
                    result["error"] = "SageAttention requires Triton, but the triton module is not available."
                else:
                    import triton

                    result["triton_version"] = str(getattr(triton, "__version__", "") or "")
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
    if modules["sageattention"]["available"]:
        recommended_attention = "sage"
    elif modules["xformers"]["available"]:
        recommended_attention = "auto"
    elif modules["flash_attn"]["available"]:
        recommended_attention = "flash"
    else:
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
            "enable_sage_attention": modules["sageattention"]["available"],
            "enable_flash_attention": modules["flash_attn"]["available"],
            "acceleration_profile_version": 2,
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
settings.input_dir.mkdir(parents=True, exist_ok=True)
app.mount("/inputs", StaticFiles(directory=settings.input_dir), name="inputs")
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
    workflow_registry.ensure_model3d_workflow_aliases()


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


@app.patch("/api/config")
async def update_config(request: SettingsUpdate) -> dict[str, Any]:
    previous_models_dir = settings.models_dir
    previous_custom_nodes_dir = settings.custom_nodes_dir
    path_fields = ("models_dir", "custom_nodes_dir", "workflows_dir")
    for field in path_fields:
        value = getattr(request, field)
        if value:
            setattr(settings, field, Path(value))

    list_path_fields = (
        "reference_model_sources",
        "reference_custom_node_sources",
        "reference_workflow_sources",
    )
    for field in list_path_fields:
        value = getattr(request, field)
        if value is not None:
            setattr(settings, field, coerce_path_list(value))

    if request.model_sources is not None:
        next_model_sources = {
            str(key): list(value)
            for key, value in settings.model_sources.items()
        }
        for key, value in request.model_sources.items():
            clean_key = str(key).strip()
            if clean_key:
                next_model_sources[clean_key] = [Path(item) for item in value if str(item).strip()]
        for key, value in DEFAULT_MODEL_SOURCES.items():
            next_model_sources.setdefault(key, list(value))
        settings.model_sources = next_model_sources

    if request.runtime is not None:
        _apply_runtime_options(request.runtime)

    settings.ensure_directories()
    save_settings(settings)
    custom_nodes_changed = request.custom_nodes_dir is not None and settings.custom_nodes_dir != previous_custom_nodes_dir
    if (
        request.models_dir is not None
        or request.model_sources is not None
        or request.reference_model_sources is not None
        or custom_nodes_changed
        or request.reference_custom_node_sources is not None
    ):
        sync_startup_model_path(settings, previous_models_dir=previous_models_dir)
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


def _trellis2_model_dir() -> Path:
    return settings.models_dir / "3d" / "trellis2" / "TRELLIS.2-4B"


def _huggingface_token() -> str | None:
    for key in ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key, "").strip()
        if value:
            return value
    if HF_TOKEN_PATH.exists():
        for line in HF_TOKEN_PATH.read_text(encoding="utf-8-sig").splitlines():
            value = line.strip()
            if value and not value.startswith("#"):
                return value
    return None


def _dinov3_model_dir() -> Path:
    return settings.models_dir / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"


def _dinov3_candidate_dirs() -> list[Path]:
    roots: list[Path] = [settings.models_dir]
    roots.extend(settings.model_sources.get("3d", []))
    roots.extend(settings.model_sources.get("clip_vision", []))
    roots.extend(settings.reference_model_sources)
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for candidate in (
            root / "facebook" / "dinov3-vitl16-pretrain-lvd1689m",
            root / "dinov3-vitl16-pretrain-lvd1689m",
        ):
            key = str(candidate)
            if key.lower() not in seen:
                seen.add(key.lower())
                candidates.append(candidate)
    return candidates


def _dinov3_installed_dir() -> Path:
    for candidate in _dinov3_candidate_dirs():
        if _dinov3_snapshot_files(candidate):
            return candidate
    return _dinov3_model_dir()


def _dinov3_snapshot_files(target: Path) -> list[str]:
    if not target.exists():
        return []
    required = target / "model.safetensors"
    if not required.exists() or required.stat().st_size <= 0:
        return []
    files: list[str] = []
    for path in target.rglob("*"):
        if not path.is_file() or ".cache" in path.parts:
            continue
        if path.stat().st_size <= 0:
            continue
        files.append(path.relative_to(target).as_posix())
    return sorted(files)


def _copy_dinov3_snapshot(source_dir: Path, target_dir: Path) -> None:
    source_root = source_dir
    nested = source_dir / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
    if (nested / "model.safetensors").exists():
        source_root = nested
    if not (source_root / "model.safetensors").exists():
        raise RuntimeError(f"Kaggle DINOv3 download did not contain model.safetensors under {source_dir}")
    target_dir.mkdir(parents=True, exist_ok=True)
    for item in source_root.iterdir():
        target = target_dir / item.name
        if item.is_dir():
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def _trellis2_candidate_dirs() -> list[Path]:
    roots: list[Path] = [settings.models_dir]
    roots.extend(settings.model_sources.get("3d", []))
    roots.extend(settings.reference_model_sources)
    candidates: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        for candidate in (
            root / "3d" / "trellis2" / "TRELLIS.2-4B",
            root / "trellis2" / "TRELLIS.2-4B",
            root / "microsoft" / "TRELLIS.2-4B",
            root / "TRELLIS.2-4B",
        ):
            key = str(candidate)
            if key.lower() not in seen:
                seen.add(key.lower())
                candidates.append(candidate)
    return candidates


def _trellis2_installed_dir() -> Path:
    for candidate in _trellis2_candidate_dirs():
        if _trellis2_snapshot_files(candidate):
            return candidate
    return _trellis2_model_dir()


def _trellis2_snapshot_files(target: Path) -> list[str]:
    if not target.exists():
        return []
    files: list[str] = []
    for path in target.rglob("*"):
        if not path.is_file() or ".cache" in path.parts:
            continue
        if path.stat().st_size <= 0:
            continue
        files.append(path.relative_to(target).as_posix())
    return sorted(files)


@app.get("/api/model3d/trellis2/status")
async def model3d_trellis2_status() -> dict[str, Any]:
    target = _trellis2_installed_dir()
    existing = _trellis2_snapshot_files(target)
    return {
        "installed": bool(existing),
        "path": str(target),
        "existing": existing,
        "missing": [] if existing else ["snapshot"],
        "source": f"https://huggingface.co/{TRELLIS2_REPO_ID}",
    }


@app.get("/api/model3d/dinov3/status")
async def model3d_dinov3_status() -> dict[str, Any]:
    target = _dinov3_installed_dir()
    existing = _dinov3_snapshot_files(target)
    return {
        "installed": bool(existing),
        "path": str(target),
        "existing": existing,
        "missing": [] if existing else ["model.safetensors"],
        "source": f"https://huggingface.co/{DINOV3_REPO_ID}",
        "token_file": str(HF_TOKEN_PATH),
        "token_configured": bool(_huggingface_token()),
    }


MODEL3D_REQUIRED_NODE_CLASSES = {
    "trellis2loadmodel",
    "trellis2loadimagewithtransparency",
    "trellis2preprocessimage",
    "trellis2imagecondmultiviewgenerator",
    "trellis2sparsemultiviewgenerator",
    "trellis2shapemultiviewgenerator",
    "trellis2shapecascademultiviewgenerator",
    "trellis2texslatmultiviewgenerator",
    "trellis2reconstructmeshwithquad",
    "trellis2simplifymesh",
    "trellis2fillholesnicelywithmeshlib",
    "trellis2fillholeswithcumesh",
    "trellis2decodelatents",
    "trellis2unwrapandrasterizer",
    "trellis2exportmesh",
    "preview3d",
}


def _model3d_node_status(object_info: dict[str, Any]) -> dict[str, Any]:
    if not object_info:
        return {"checked": False, "available": [], "missing": []}
    available = {str(key).lower() for key in (object_info or {}).keys()}
    missing = sorted(name for name in MODEL3D_REQUIRED_NODE_CLASSES if name not in available)
    return {
        "checked": bool(object_info),
        "available": sorted(MODEL3D_REQUIRED_NODE_CLASSES - set(missing)),
        "missing": missing,
    }


async def _model3d_preflight_report(
    *,
    start_comfy: bool = False,
    full: bool = False,
    requested_model: str = "",
) -> dict[str, Any]:
    trellis2 = await model3d_trellis2_status()
    dinov3 = await model3d_dinov3_status()
    capabilities = _runtime_attention_capabilities()
    comfy_running = await comfy.is_running()
    object_info: dict[str, Any] = {}
    workflow_analysis: dict[str, Any] = {}
    object_info_error = ""

    if full or start_comfy:
        try:
            if start_comfy:
                await comfy.ensure_running()
                comfy_running = True
            if comfy_running:
                object_info = await comfy.object_info()
        except Exception as exc:
            object_info_error = f"{type(exc).__name__}: {str(exc)[:240]}"

    node_status = _model3d_node_status(object_info)
    workflow_registry.ensure_model3d_workflow_aliases()
    workflow_path = workflow_registry.find("model3d-trellis2-meshwithvoxel-texturing-multiview", "Model3D")
    workflow_id = "model3d-trellis2-meshwithvoxel-texturing-multiview"
    if workflow_path is None:
        workflow_path = workflow_registry.find("model3d-trellis2-meshwithtexturing-multiview", "Model3D")
        workflow_id = "model3d-trellis2-meshwithtexturing-multiview"
    if workflow_path and object_info:
        try:
            workflow = workflow_registry.summarize(workflow_path)
            workflow_analysis = workflow_registry.analyze_workflow(workflow, object_info=object_info).model_dump(mode="json")
        except Exception as exc:
            workflow_analysis = {"error": f"{type(exc).__name__}: {str(exc)[:240]}"}

    blocking: list[str] = []
    warnings: list[str] = []
    if not trellis2.get("installed"):
        blocking.append("TRELLIS.2-4B checkpoint is missing.")
    if not dinov3.get("installed"):
        blocking.append("DINOv3 ViT-L/16 model is missing.")
    if not capabilities.get("torch", {}).get("cuda_available"):
        blocking.append("CUDA GPU is not available for 3D generation.")
    if node_status.get("checked") and node_status.get("missing"):
        blocking.append("Required TRELLIS.2 custom nodes are missing: " + ", ".join(node_status["missing"][:6]) + ".")
    if object_info_error:
        warnings.append(f"ComfyUI object registry could not be read: {object_info_error}")
    if not workflow_path:
        blocking.append("Model 3D workflow is missing: workflows/comfyui/model3d_trellis2_meshwithvoxel_texturing_multiview.json.")
    if not node_status.get("checked") and not full:
        warnings.append("Custom node registry was not checked in quick mode.")
    if not comfy_running:
        warnings.append("ComfyUI is not running; it will start on demand.")

    active_signature = list(last_generation_model_signature) if last_generation_model_signature else []
    requested = requested_model.strip()
    preloaded_match = bool(active_signature and requested and requested in active_signature[-1])
    return {
        "ok": not blocking,
        "mode": "full" if full or start_comfy else "quick",
        "blocking": blocking,
        "warnings": warnings,
        "trellis2": trellis2,
        "dinov3": dinov3,
        "runtime": capabilities,
        "comfy_running": comfy_running,
        "object_registry_checked": bool(object_info),
        "object_registry_error": object_info_error,
        "required_nodes": node_status,
        "workflow": {
            "id": workflow_id,
            "path": str(workflow_path) if workflow_path else "",
            "analysis": workflow_analysis,
        },
        "preloaded": {
            "last_generation_model_signature": active_signature,
            "requested_model": requested,
            "matches_last_loaded_signature": preloaded_match,
            "will_clear_previous_model": bool(active_signature and requested and not preloaded_match),
        },
    }


@app.get("/api/model3d/preflight")
async def model3d_preflight(
    start_comfy: bool = Query(False),
    full: bool = Query(False),
    requested_model: str = Query(""),
) -> dict[str, Any]:
    report = await _model3d_preflight_report(
        start_comfy=start_comfy,
        full=full,
        requested_model=requested_model,
    )
    if report.get("blocking"):
        detail = " ".join(str(item) for item in report.get("blocking", []))
        print(f"NEXUS BTA WARN Model 3D preflight: {detail}", flush=True)
    if report.get("required_nodes", {}).get("missing") and (full or start_comfy):
        missing = ", ".join(report["required_nodes"]["missing"][:8])
        print(f"NEXUS BTA WARN Model 3D custom node check: missing {missing}", flush=True)
    return report


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


@app.post("/api/custom-nodes/install-missing")
async def install_missing_custom_nodes(
    request: DependencyInstallRequest,
    restart_comfy: bool = Query(True),
) -> dict[str, Any]:
    missing_nodes = [str(item).strip() for item in request.node_names if str(item).strip()]
    if not missing_nodes:
        raise HTTPException(status_code=400, detail="Missing node class list is empty.")

    suggestions = manager_suggestions_for_nodes(settings, missing_nodes)
    targets = custom_nodes_for_workflow(settings, missing_nodes, suggestions)
    if not targets:
        return {
            "installed": [],
            "updated": [],
            "errors": {},
            "targets": [],
            "suggestions": suggestions,
            "message": "No install target was found for the missing nodes. Open Manager and install the suggested custom node manually.",
        }

    updated: list[dict[str, Any]] = []
    update_errors: dict[str, str] = {}
    installed_node_names = {node.name.lower(): node for node in scan_custom_nodes(settings)}
    for target in targets:
        if str(target).lower().startswith(("http://", "https://", "git@")):
            continue
        node = installed_node_names.get(str(target).lower())
        if not node:
            continue
        node_path = Path(node.path)
        if not (node_path / ".git").exists():
            continue
        try:
            updated.append(_update_custom_node(node_path, ""))
        except Exception as exc:
            update_errors[str(target)] = str(getattr(exc, "detail", exc))[-1200:]

    installed, errors = install_custom_node_dependencies(
        settings,
        node_names=targets,
        all_enabled=False,
    )
    errors.update(update_errors)

    restarted = False
    if restart_comfy and (installed or updated) and await comfy.is_running():
        cleanup_embedded_comfy_artifacts()
        await comfy.restart()
        restarted = True

    object_info = await _optional_comfy_object_info()
    still_missing = [node for node in missing_nodes if not _available_comfy_node(object_info, node)]
    return {
        "installed": installed,
        "updated": updated,
        "errors": errors,
        "targets": targets,
        "suggestions": suggestions,
        "restarted_comfy": restarted,
        "missing_after_install": still_missing,
        "status": custom_node_dependency_status(settings),
    }


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
    result = _update_custom_node(path, request.version)
    installed: list[str] = []
    errors: dict[str, str] = {}
    if request.install_dependencies:
        installed, errors = install_custom_node_dependencies(settings, node_names=[path.name], all_enabled=False)
    return {**result, "installed_dependencies": installed, "dependency_errors": errors}


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
    workflow_registry.ensure_model3d_workflow_aliases()
    workflow_path = workflow_registry.find(workflow_id, "Model3D" if workflow_id.startswith("model3d-") else None)
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
    workflow_registry.ensure_model3d_workflow_aliases()
    workflow_path = workflow_registry.find(workflow_id, "Model3D" if workflow_id.startswith("model3d-") else None)
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
async def start_comfy(wait: bool = Query(True)) -> dict[str, Any]:
    try:
        cleanup_embedded_comfy_artifacts()
        if not wait:
            return await comfy.start_nowait()
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


@app.post("/api/backend/restart")
async def restart_backend(delay: float = Query(2.0)) -> dict[str, Any]:
    python = sys.executable or "python"
    runner = settings.project_root / "backend" / "run_backend.py"
    if not runner.exists():
        raise HTTPException(status_code=404, detail=f"Backend runner not found: {runner}")
    delay_value = max(0.5, min(15.0, float(delay or 2.0)))
    helper = (
        "import subprocess,time,os;"
        f"time.sleep({delay_value!r});"
        f"subprocess.Popen({[python, str(runner)]!r}, cwd={str(settings.project_root)!r});"
    )
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
    subprocess.Popen([python, "-c", helper], cwd=settings.project_root, creationflags=creationflags)
    async def _exit_soon() -> None:
        await asyncio.sleep(0.25)
        os._exit(0)

    asyncio.create_task(_exit_soon())
    return {"status": "restarting", "delay_seconds": delay_value}


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


def _download_url_to_file(url: str, target: Path, job_id: str) -> dict[str, Any]:
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(target.suffix + ".part")
    downloaded = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "NexusBTA/1.0"}
    if downloaded:
        headers["Range"] = f"bytes={downloaded}-"
    request = Request(url, headers=headers)
    started = time.monotonic()
    with urlopen(request, timeout=60) as response:
        length_header = response.headers.get("Content-Length")
        remaining = int(length_header) if length_header and str(length_header).isdigit() else 0
        total = downloaded + remaining if remaining else 0
        mode = "ab" if downloaded and response.status == 206 else "wb"
        if mode == "wb":
            downloaded = 0
        last_emit = 0.0
        with partial.open(mode + "") as handle:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                handle.write(chunk)
                downloaded += len(chunk)
                now = time.monotonic()
                if now - last_emit >= 0.5:
                    _update_download_job(
                        job_id,
                        {
                            "status": "downloading",
                            "bytes_downloaded": downloaded,
                            "bytes_total": total,
                            "progress": round((downloaded / total) * 100, 2) if total else None,
                            "speed_bps": round(downloaded / max(0.001, now - started)),
                            "message": f"Downloading {target.name}",
                        },
                    )
                    last_emit = now
    partial.replace(target)
    return {
        "status": "downloaded",
        "filename": target.name,
        "path": str(target),
        "relative_path": _download_relative_path(target),
        "bytes_downloaded": target.stat().st_size,
        "bytes_total": target.stat().st_size,
        "progress": 100,
    }


def _ideogram4_artifact_target(key: str) -> Path:
    artifact = IDEOGRAM4_HF_ARTIFACTS[key]
    parts = [str(part) for part in artifact["target"]]
    return settings.models_dir.joinpath(*parts)


def _ideogram4_artifact_status(key: str) -> dict[str, Any]:
    artifact = IDEOGRAM4_HF_ARTIFACTS[key]
    target = _ideogram4_artifact_target(key)
    min_bytes = int(artifact.get("min_bytes") or 1024 * 1024)
    installed = target.exists() and target.stat().st_size >= min_bytes
    return {
        "key": key,
        "label": artifact["label"],
        "filename": artifact["filename"],
        "url": artifact["url"],
        "kind": artifact.get("kind") or "model",
        "scope": artifact.get("scope") or "dependency",
        "destination": str(target),
        "path": str(target) if installed else "",
        "installed": bool(installed),
        "size_bytes_min": min_bytes,
        "size_bytes": target.stat().st_size if target.exists() else min_bytes,
    }


def _ideogram4_missing_core_support(object_info: dict[str, Any] | None) -> list[str]:
    registry = object_info or {}
    missing = [name for name in IDEOGRAM4_REQUIRED_COMFY_NODES if name not in registry]
    clip_info = registry.get("CLIPLoader") or {}
    clip_type_options = (
        ((clip_info.get("input") or {}).get("required") or {}).get("type") or [[], {}]
    )
    try:
        clip_types = set(str(item) for item in (clip_type_options[0] or []))
    except (TypeError, IndexError):
        clip_types = set()
    if "ideogram4" not in clip_types:
        missing.append("CLIPLoader type ideogram4")
    return missing


async def _ideogram4_status_snapshot() -> dict[str, Any]:
    assets = [_ideogram4_artifact_status(key) for key in IDEOGRAM4_HF_ARTIFACTS]
    required_keys = {"checkpoint", "unconditional_checkpoint", "qwen3vl", "vae"}
    missing_required = [item for item in assets if item["key"] in required_keys and not item["installed"]]
    missing_optional = [item for item in assets if item["key"] not in required_keys and not item["installed"]]
    dependency_status = custom_node_dependency_status(settings)
    relevant_nodes = {
        name: status
        for name, status in dependency_status.items()
        if name.lower() in {"comfyui-kjnodes", "rgthree-comfy", "res4lyf", "comfymath"}
    }
    missing_node_dependencies = [name for name, status in relevant_nodes.items() if not status.get("installed")]
    runtime_checked = await comfy.is_running()
    missing_core_nodes: list[str] = []
    if runtime_checked:
        try:
            missing_core_nodes = _ideogram4_missing_core_support(await comfy.object_info())
        except Exception as exc:
            missing_core_nodes = [f"Comfy object_info unavailable: {exc}"]
    return {
        "template": "Ideogram4",
        "label": "Ideogram 4",
        "installed": not missing_required,
        "generation_ready": not missing_required and not missing_core_nodes,
        "dependencies_installed": not missing_required and not missing_node_dependencies and not missing_core_nodes,
        "assets": assets,
        "missing_assets": missing_required + missing_optional,
        "missing_required_assets": missing_required,
        "missing_optional_assets": missing_optional,
        "custom_node_dependencies": relevant_nodes,
        "missing_custom_node_dependencies": missing_node_dependencies,
        "runtime_checked": runtime_checked,
        "missing_core_nodes": missing_core_nodes,
        "models_dir": str(settings.models_dir),
        "estimated_missing_required_bytes": sum(int(item.get("size_bytes_min") or 0) for item in missing_required),
        "estimated_missing_optional_bytes": sum(int(item.get("size_bytes_min") or 0) for item in missing_optional),
        "restart_recommended": bool(missing_core_nodes),
        "note": "Ideogram 4 downloads are optional for new users. Generation requires the main FP8 model, unconditional FP8 model, Qwen3-VL text encoder and Flux2 VAE. Gemma 4 is optional for prompt-helper workflows.",
    }


async def _run_ideogram4_assets_download_job(job_id: str, keys: list[str] | None = None, install_node_dependencies: bool = False) -> None:
    try:
        selected_keys = [key for key in (keys or []) if key in IDEOGRAM4_HF_ARTIFACTS]
        if not selected_keys:
            selected_keys = [item["key"] for item in (await _ideogram4_status_snapshot())["missing_required_assets"]]
        if not selected_keys and not install_node_dependencies:
            raise ValueError("No valid Ideogram 4 dependency assets selected.")
        completed: list[dict[str, Any]] = []
        total = max(1, len(selected_keys) + (1 if install_node_dependencies else 0))
        step = 0
        for key in selected_keys:
            step += 1
            status = _ideogram4_artifact_status(key)
            if status["installed"]:
                completed.append({**status, "already_downloaded": True})
                _update_download_job(job_id, {"message": f"Ideogram 4 asset already present: {status['filename']}", "progress": round((step / total) * 100, 2)})
                continue
            artifact = IDEOGRAM4_HF_ARTIFACTS[key]
            target = _ideogram4_artifact_target(key)
            _update_download_job(job_id, {"status": "downloading", "message": f"Downloading {artifact['label']}: {artifact['filename']}"})
            result = await asyncio.to_thread(_download_url_to_file, str(artifact["url"]), target, job_id)
            completed.append({**result, "key": key, "label": artifact["label"]})
        dependency_errors: dict[str, str] = {}
        dependencies_installed: list[str] = []
        if install_node_dependencies:
            step += 1
            _update_download_job(job_id, {"message": "Installing Ideogram 4 custom-node Python dependencies.", "progress": round((step / total) * 100, 2)})
            dependencies_installed, dependency_errors = await asyncio.to_thread(
                install_custom_node_dependencies,
                settings,
                ["ComfyUI-KJNodes", "rgthree-comfy", "RES4LYF", "ComfyMath"],
                False,
            )
        ensure_model_tree(settings)
        _update_download_job(
            job_id,
            {
                "status": "downloaded",
                "progress": 100,
                "message": "Ideogram 4 selected dependencies ready.",
                "assets": completed,
                "dependencies_installed": dependencies_installed,
                "dependency_errors": dependency_errors,
                "status_snapshot": await _ideogram4_status_snapshot(),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/ideogram4/assets/status")
async def ideogram4_assets_status() -> dict[str, Any]:
    return await _ideogram4_status_snapshot()


@app.post("/api/ideogram4/assets/download/start")
async def ideogram4_assets_download_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    raw_keys = payload.get("assets")
    selected_keys = [str(item).strip() for item in raw_keys] if isinstance(raw_keys, list) else None
    install_node_dependencies = bool(payload.get("install_node_dependencies"))
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "kind": "ideogram4_assets",
        "status": "queued",
        "progress": 0,
        "message": "Queued Ideogram 4 optional dependency download.",
        "assets": selected_keys or [],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ideogram4_assets_download_job(job_id, selected_keys, install_node_dependencies))
    return download_jobs[job_id]


def _nvidia_pid_dir() -> Path:
    return settings.custom_nodes_dir / "ComfyUI-PiD" / "vendor" / "PiD"


def _nvidia_pid_source_ready() -> bool:
    return (_nvidia_pid_dir() / "pid" / "_src" / "utils" / "model_loader.py").is_file()


def _wan22_artifact_target(key: str) -> Path:
    artifact = WAN22_HF_ARTIFACTS[key]
    category, filename = artifact["target"]
    return settings.models_dir / str(category) / str(filename)


def _controlnet_artifact_target(key: str) -> Path:
    artifact = CONTROLNET_OPTIONAL_ARTIFACTS[key]
    parts = [str(part) for part in artifact["target"]]
    return settings.models_dir.joinpath(*parts)


def _controlnet_artifact_installed_path(key: str) -> Path:
    artifact = CONTROLNET_OPTIONAL_ARTIFACTS[key]
    target = _controlnet_artifact_target(key)
    min_bytes = int(artifact.get("min_bytes") or 1024 * 1024)
    candidates = [target]
    filename = str(artifact.get("filename") or "")
    if filename:
        candidates.extend(settings.models_dir.glob(f"controlnet/**/{filename}"))
        candidates.extend(settings.models_dir.glob(f"model_patches/**/{filename}"))
    if key == "qwen_union":
        candidates.extend(settings.models_dir.glob("controlnet/**/*InstantX*ControlNet*Union*.safetensors"))
        candidates.extend(settings.models_dir.glob("controlnet/**/*Qwen*ControlNet*Union*.safetensors"))
    if key == "flux_union":
        candidates.extend(settings.models_dir.glob("controlnet/**/*Flux*ControlNet*Union*.safetensors"))
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.stat().st_size >= min_bytes:
                return candidate
        except OSError:
            continue
    return target


def _controlnet_artifact_status(key: str) -> dict[str, Any]:
    artifact = CONTROLNET_OPTIONAL_ARTIFACTS[key]
    target = _controlnet_artifact_installed_path(key)
    installed = target.exists() and target.stat().st_size >= int(artifact.get("min_bytes") or 1024 * 1024)
    canonical = _controlnet_artifact_target(key)
    return {
        "key": key,
        "label": artifact["label"],
        "preset": artifact["preset"],
        "types": artifact["types"],
        "filename": artifact["filename"],
        "url": artifact["url"],
        "destination": str(canonical),
        "path": str(target),
        "installed": bool(installed),
        "size_bytes": int(artifact.get("size_bytes") or 0),
        "scope": artifact.get("scope") or "dependency",
    }


def _controlnet_status_snapshot(preset: str | None = None, control_type: str | None = None) -> dict[str, Any]:
    preset_norm = str(preset or "").strip().lower()
    type_norm = str(control_type or "").strip().lower()
    assets = []
    for key in CONTROLNET_OPTIONAL_ARTIFACTS:
        status = _controlnet_artifact_status(key)
        if preset_norm and str(status["preset"]).lower() != preset_norm and not (
            preset_norm in {"sd15", "sd1.5"} and status["preset"] == "SD"
        ) and not (
            preset_norm in {"sdxl", "xl"} and status["preset"] == "XL"
        ):
            continue
        if type_norm and type_norm not in {str(item).lower() for item in status["types"]}:
            continue
        assets.append(status)
    missing = [item for item in assets if not item["installed"]]
    return {
        "installed": len(missing) == 0 if assets else False,
        "assets": assets,
        "missing_assets": missing,
        "models_dir": str(settings.models_dir),
        "estimated_missing_bytes": sum(int(item.get("size_bytes") or 0) for item in missing),
        "note": "ControlNet models are optional dependency assets. Nexus downloads them only after UI confirmation and does not modify Torch/CUDA requirements.",
    }


def _wan22_artifact_installed_path(key: str) -> Path:
    target = _wan22_artifact_target(key)
    min_bytes = int(WAN22_HF_ARTIFACTS[key].get("min_bytes") or 1024 * 1024)
    candidates = [target]
    if key == "vae22":
        candidates.append(settings.models_dir / "vae" / "wan22-vae.safetensors")
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size >= min_bytes:
            return candidate
    return target


def _wan22_artifact_status(key: str) -> dict[str, Any]:
    artifact = WAN22_HF_ARTIFACTS[key]
    target = _wan22_artifact_installed_path(key)
    min_bytes = int(artifact.get("min_bytes") or 1024 * 1024)
    installed = target.exists() and target.stat().st_size >= min_bytes
    return {
        "key": key,
        "label": artifact["label"],
        "filename": artifact["filename"],
        "url": artifact["url"],
        "kind": artifact.get("kind") or "model",
        "scope": artifact.get("scope") or "dependency",
        "destination": str(_wan22_artifact_target(key)),
        "path": str(target) if installed else "",
        "installed": installed,
        "size_bytes_min": min_bytes,
        "source": "Hugging Face",
    }


def _wan_motion_artifact_target(key: str) -> Path:
    artifact = WAN_MOTION_CAPTURE_ARTIFACTS[key]
    parts = [str(part) for part in artifact["target"]]
    return settings.models_dir.joinpath(*parts)


def _wan_motion_artifact_installed_path(key: str) -> Path:
    artifact = WAN_MOTION_CAPTURE_ARTIFACTS[key]
    target = _wan_motion_artifact_target(key)
    min_bytes = int(artifact.get("min_bytes") or 1024 * 1024)
    candidates = [target]
    filename = str(artifact.get("filename") or "")
    if filename:
        candidates.extend(settings.models_dir.glob(f"**/{filename}"))
    for candidate in candidates:
        try:
            if candidate.exists() and candidate.stat().st_size >= min_bytes:
                return candidate
        except OSError:
            continue
    return target


def _wan_motion_artifact_status(key: str) -> dict[str, Any]:
    artifact = WAN_MOTION_CAPTURE_ARTIFACTS[key]
    target = _wan_motion_artifact_installed_path(key)
    min_bytes = int(artifact.get("min_bytes") or 1024 * 1024)
    installed = target.exists() and target.stat().st_size >= min_bytes
    canonical = _wan_motion_artifact_target(key)
    return {
        "key": key,
        "label": artifact["label"],
        "filename": artifact["filename"],
        "url": artifact["url"],
        "scope": artifact.get("scope") or "dependency",
        "destination": str(canonical),
        "path": str(target) if installed else "",
        "installed": bool(installed),
        "size_bytes_min": min_bytes,
        "size_bytes": int(artifact.get("size_bytes") or min_bytes),
        "source": artifact.get("source") or "source",
    }


def _wan_motion_control_model_status() -> dict[str, Any]:
    roots = [
        settings.models_dir / "checkpoints",
        settings.models_dir / "diffusion_models",
        settings.models_dir / "unet",
    ]
    high_candidates: list[dict[str, Any]] = []
    low_candidates: list[dict[str, Any]] = []
    generic_candidates: list[dict[str, Any]] = []
    suffixes = {".safetensors", ".gguf", ".pt", ".pth"}
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            name = path.name.lower()
            if "wan" not in name:
                continue
            is_fun_control = "fun" in name and "control" in name
            is_animate = "animate" in name
            if not (is_fun_control or is_animate):
                continue
            item = {
                "name": str(path.relative_to(settings.models_dir)).replace("\\", "/"),
                "filename": path.name,
                "size_bytes": path.stat().st_size,
            }
            if "high" in name:
                high_candidates.append(item)
            elif "low" in name:
                low_candidates.append(item)
            else:
                generic_candidates.append(item)
    single_file_ready = bool(generic_candidates)
    return {
        "ready": bool((high_candidates and low_candidates) or single_file_ready),
        "high_candidates": high_candidates,
        "low_candidates": low_candidates,
        "generic_candidates": generic_candidates,
        "single_file_ready": single_file_ready,
        "note": "WAN Motion Capture generation needs compatible high/low WAN Fun-Control or Animate motion-control models, or a compatible single-file WAN Fun-Control 5B/GGUF base model. LoRAs can help speed/style but do not replace that route.",
    }


def _wan_motion_custom_node_status(key: str) -> dict[str, Any]:
    node = WAN_MOTION_CAPTURE_CUSTOM_NODES[key]
    path = settings.custom_nodes_dir / node["folder"]
    return {
        "key": key,
        "label": node["label"],
        "repo": node["repo"],
        "destination": str(path),
        "installed": path.exists(),
        "source": node.get("source") or "GitHub",
    }


def _wan_motion_missing_node_groups(object_info: dict[str, Any]) -> list[dict[str, Any]]:
    missing: list[dict[str, Any]] = []
    for key, names in WAN_MOTION_CAPTURE_NODE_GROUPS.items():
        if not _available_comfy_node(object_info, *names):
            missing.append({"key": key, "accepted_nodes": list(names), "label": names[0]})
    return missing


async def _wan_motion_status_snapshot() -> dict[str, Any]:
    object_info = await _optional_comfy_object_info()
    node_groups_missing = _wan_motion_missing_node_groups(object_info)
    custom_nodes = [_wan_motion_custom_node_status(key) for key in WAN_MOTION_CAPTURE_CUSTOM_NODES]
    artifacts = [_wan_motion_artifact_status(key) for key in WAN_MOTION_CAPTURE_ARTIFACTS]
    dependency_assets = [item for item in artifacts if item.get("scope") != "base_model"]
    base_assets = [item for item in artifacts if item.get("scope") == "base_model"]
    missing_custom_nodes = [item for item in custom_nodes if not item["installed"]]
    missing_dependencies = [item for item in dependency_assets if not item["installed"]]
    missing_base = [item for item in base_assets if not item["installed"]]
    ready_dependencies = not missing_custom_nodes and not missing_dependencies and not node_groups_missing
    motion_control_models = _wan_motion_control_model_status()
    return {
        "template": "Wan",
        "label": "WAN Motion Capture",
        "installed": ready_dependencies,
        "dependencies_installed": ready_dependencies,
        "generation_ready": bool(ready_dependencies and motion_control_models["ready"]),
        "motion_control_models_ready": bool(motion_control_models["ready"]),
        "motion_control_models": motion_control_models,
        "custom_nodes": custom_nodes,
        "missing_custom_nodes": missing_custom_nodes,
        "node_groups_missing": node_groups_missing,
        "assets": artifacts,
        "dependency_assets": dependency_assets,
        "base_model_assets": base_assets,
        "missing_assets": missing_dependencies,
        "missing_dependency_assets": missing_dependencies,
        "missing_base_model_assets": missing_base,
        "models_dir": str(settings.models_dir),
        "custom_nodes_dir": str(settings.custom_nodes_dir),
        "estimated_missing_dependency_bytes": sum(int(item.get("size_bytes") or item.get("size_bytes_min") or 0) for item in missing_dependencies),
        "estimated_missing_base_model_bytes": sum(int(item.get("size_bytes") or item.get("size_bytes_min") or 0) for item in missing_base),
        "restart_recommended": bool(missing_custom_nodes),
        "note": "WAN Motion Capture is optional. Dependencies are separate from base/checkpoint assets. DWPose/Fun-Control motion requires compatible WAN Fun-Control or WAN Animate models; Nexus does not auto-download base/checkpoint models.",
    }


def _wan22_status_snapshot() -> dict[str, Any]:
    assets = [_wan22_artifact_status(key) for key in WAN22_HF_ARTIFACTS]
    dependency_assets = [item for item in assets if item.get("scope") != "base_model"]
    base_assets = [item for item in assets if item.get("scope") == "base_model"]
    missing_dependencies = [item for item in dependency_assets if not item["installed"]]
    missing_base = [item for item in base_assets if not item["installed"]]
    ready_core = all(item["installed"] for item in assets if item["key"] in {"clip_vision", "umt5", "vae21"})
    ready_i2v_full = all(item["installed"] for item in assets if item["key"] in {"clip_vision", "umt5", "vae21", "high_noise", "low_noise"})
    return {
        "template": "Wan",
        "label": "WAN 2.2",
        "models_dir": str(settings.models_dir),
        "ready_core": ready_core,
        "ready_i2v_full": ready_i2v_full,
        "installed": len(missing_dependencies) == 0,
        "dependencies_installed": len(missing_dependencies) == 0,
        "assets": assets,
        "dependency_assets": dependency_assets,
        "base_model_assets": base_assets,
        "missing_assets": missing_dependencies,
        "missing_dependency_assets": missing_dependencies,
        "missing_base_model_assets": missing_base,
        "estimated_missing_min_bytes": sum(int(item.get("size_bytes_min") or 0) for item in missing_dependencies),
        "estimated_missing_dependency_min_bytes": sum(int(item.get("size_bytes_min") or 0) for item in missing_dependencies),
        "estimated_missing_base_model_min_bytes": sum(int(item.get("size_bytes_min") or 0) for item in missing_base),
        "restart_recommended": True,
        "note": "WAN 2.2 template dependency assets are optional and downloaded only after UI confirmation. Base/checkpoint diffusion models are not treated as template dependencies.",
    }


async def _run_wan22_assets_download_job(job_id: str, keys: list[str] | None = None) -> None:
    try:
        selected_keys = [key for key in (keys or list(WAN22_HF_ARTIFACTS)) if key in WAN22_HF_ARTIFACTS]
        if not selected_keys:
            raise ValueError("No valid WAN 2.2 dependency assets selected.")
        _update_download_job(job_id, {"status": "downloading", "progress": 0, "message": "Preparing WAN 2.2 dependency asset downloads."})
        completed = []
        total = len(selected_keys)
        for index, key in enumerate(selected_keys, start=1):
            status = _wan22_artifact_status(key)
            if status["installed"]:
                completed.append({**status, "already_downloaded": True})
                _update_download_job(job_id, {"message": f"WAN asset already present: {status['filename']}", "progress": round((index / total) * 100, 2)})
                continue
            artifact = WAN22_HF_ARTIFACTS[key]
            target = _wan22_artifact_target(key)
            _update_download_job(job_id, {"message": f"Downloading {artifact['label']}: {artifact['filename']}"})
            result = await asyncio.to_thread(_download_url_to_file, str(artifact["url"]), target, job_id)
            completed.append({**result, "key": key, "label": artifact["label"]})
        ensure_model_tree(settings)
        _update_download_job(
            job_id,
            {
                "status": "downloaded",
                "progress": 100,
                "message": "WAN 2.2 dependency assets ready.",
                "assets": completed,
                "status_snapshot": _wan22_status_snapshot(),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/wan22/assets/status")
async def wan22_assets_status() -> dict[str, Any]:
    return _wan22_status_snapshot()


@app.post("/api/wan22/assets/download/start")
async def wan22_assets_download_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = payload.get("assets") if isinstance(payload, dict) else None
    selected_keys = [str(item).strip().lower() for item in selected] if isinstance(selected, list) else None
    if selected_keys is None:
        selected_keys = [item["key"] for item in _wan22_status_snapshot()["missing_dependency_assets"]]
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "kind": "wan22_assets",
        "status": "queued",
        "progress": 0,
        "message": "Queued WAN 2.2 optional dependency asset download.",
        "assets": selected_keys,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_wan22_assets_download_job(job_id, selected_keys))
    return download_jobs[job_id]


async def _run_wan_motion_capture_download_job(
    job_id: str,
    keys: list[str] | None = None,
    *,
    install_nodes: bool = True,
    include_base_model: bool = False,
) -> None:
    try:
        selected_keys = [key for key in (keys or list(WAN_MOTION_CAPTURE_ARTIFACTS)) if key in WAN_MOTION_CAPTURE_ARTIFACTS]
        if not include_base_model:
            selected_keys = [
                key
                for key in selected_keys
                if WAN_MOTION_CAPTURE_ARTIFACTS[key].get("scope") != "base_model"
            ]
        _update_download_job(job_id, {"status": "downloading", "progress": 0, "message": "Preparing WAN Motion Capture optional dependencies."})
        completed_nodes: list[dict[str, Any]] = []
        errors: dict[str, str] = {}
        if install_nodes:
            root = settings.custom_nodes_dir.resolve()
            root.mkdir(parents=True, exist_ok=True)
            total_nodes = max(1, len(WAN_MOTION_CAPTURE_CUSTOM_NODES))
            for index, node in enumerate(WAN_MOTION_CAPTURE_CUSTOM_NODES.values(), start=1):
                target = (root / node["folder"]).resolve()
                if not target.is_relative_to(root):
                    errors[node["folder"]] = "Invalid custom node destination."
                    continue
                _update_download_job(
                    job_id,
                    {
                        "message": f"Preparing {node['label']}",
                        "progress": round((index / (total_nodes + max(1, len(selected_keys)))) * 40, 2),
                    },
                )
                try:
                    if target.exists() and (target / ".git").exists():
                        result = await asyncio.to_thread(_run_git, ["pull", "--ff-only"], target, 600)
                    elif target.exists():
                        completed_nodes.append({"label": node["label"], "path": str(target), "already_present": True})
                        continue
                    else:
                        result = await asyncio.to_thread(_run_git, ["clone", node["repo"], str(target)], root, 900)
                    if result.returncode != 0:
                        raise RuntimeError((result.stderr or result.stdout or "git operation failed").strip()[-1200:])
                    completed_nodes.append({"label": node["label"], "path": str(target), "repo": node["repo"]})
                except Exception as exc:
                    errors[node["folder"]] = str(exc)[-1200:]
            try:
                installed, dep_errors = await asyncio.to_thread(
                    install_custom_node_dependencies,
                    settings,
                    node_names=[node["folder"] for node in WAN_MOTION_CAPTURE_CUSTOM_NODES.values()],
                    all_enabled=False,
                )
                if installed:
                    completed_nodes.append({"label": "Python requirements", "installed_dependencies": installed})
                errors.update(dep_errors)
            except Exception as exc:
                errors["python_requirements"] = str(exc)[-1200:]

        completed_assets = []
        total_assets = len(selected_keys)
        for index, key in enumerate(selected_keys, start=1):
            status = _wan_motion_artifact_status(key)
            if status["installed"]:
                completed_assets.append({**status, "already_downloaded": True})
                _update_download_job(job_id, {"message": f"WAN Motion asset already present: {status['filename']}", "progress": 45 + round((index / max(1, total_assets)) * 50, 2)})
                continue
            artifact = WAN_MOTION_CAPTURE_ARTIFACTS[key]
            target = _wan_motion_artifact_target(key)
            _update_download_job(job_id, {"message": f"Downloading {artifact['label']}: {artifact['filename']}"})
            result = await asyncio.to_thread(_download_url_to_file, str(artifact["url"]), target, job_id)
            completed_assets.append({**result, "key": key, "label": artifact["label"]})

        ensure_model_tree(settings)
        snapshot = await _wan_motion_status_snapshot()
        final_status = "downloaded" if not errors else "completed"
        _update_download_job(
            job_id,
            {
                "status": final_status,
                "progress": 100,
                "message": "WAN Motion Capture optional setup finished." if not errors else "WAN Motion Capture setup finished with warnings.",
                "custom_nodes": completed_nodes,
                "assets": completed_assets,
                "errors": errors,
                "status_snapshot": snapshot,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/wan22/motion-capture/status")
async def wan22_motion_capture_status() -> dict[str, Any]:
    return await _wan_motion_status_snapshot()


@app.post("/api/wan22/motion-capture/download/start")
async def wan22_motion_capture_download_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    raw_keys = payload.get("assets")
    install_nodes = not (payload.get("install_nodes") is False)
    include_base_model = bool(payload.get("include_base_model"))
    if isinstance(raw_keys, list) and raw_keys:
        selected_keys = [
            str(item)
            for item in raw_keys
            if str(item) in WAN_MOTION_CAPTURE_ARTIFACTS
            and (include_base_model or WAN_MOTION_CAPTURE_ARTIFACTS[str(item)].get("scope") != "base_model")
        ]
    else:
        snapshot = await _wan_motion_status_snapshot()
        selected_keys = [item["key"] for item in snapshot["missing_dependency_assets"]]
        if include_base_model:
            selected_keys.extend(item["key"] for item in snapshot["missing_base_model_assets"])
    if not selected_keys and not install_nodes:
        raise HTTPException(status_code=400, detail="No missing WAN Motion Capture dependency assets selected.")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "kind": "wan22_motion_capture",
        "status": "queued",
        "progress": 0,
        "message": "Queued WAN Motion Capture optional setup.",
        "assets": selected_keys,
        "install_nodes": install_nodes,
        "include_base_model": include_base_model,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(
        _run_wan_motion_capture_download_job(
            job_id,
            selected_keys,
            install_nodes=install_nodes,
            include_base_model=include_base_model,
        )
    )
    return download_jobs[job_id]


async def _run_controlnet_assets_download_job(job_id: str, keys: list[str]) -> None:
    try:
        selected = [key for key in keys if key in CONTROLNET_OPTIONAL_ARTIFACTS]
        if not selected:
            raise ValueError("No valid ControlNet dependency assets selected.")
        _update_download_job(job_id, {"status": "downloading", "progress": 0, "message": "Preparing ControlNet dependency downloads."})
        completed = []
        total = len(selected)
        for index, key in enumerate(selected, start=1):
            status = _controlnet_artifact_status(key)
            if status["installed"]:
                completed.append({**status, "already_downloaded": True})
                _update_download_job(job_id, {"message": f"ControlNet already present: {status['filename']}", "progress": round((index / total) * 100, 2)})
                continue
            artifact = CONTROLNET_OPTIONAL_ARTIFACTS[key]
            target = _controlnet_artifact_target(key)
            _update_download_job(job_id, {"message": f"Downloading {artifact['label']}: {artifact['filename']}"})
            result = await asyncio.to_thread(_download_url_to_file, str(artifact["url"]), target, job_id)
            completed.append({**result, "key": key, "label": artifact["label"]})
        ensure_model_tree(settings)
        _update_download_job(
            job_id,
            {
                "status": "downloaded",
                "progress": 100,
                "message": "ControlNet dependency assets ready.",
                "assets": completed,
                "status_snapshot": _controlnet_status_snapshot(),
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/controlnet/assets/status")
async def controlnet_assets_status(preset: str | None = None, type: str | None = None) -> dict[str, Any]:
    return _controlnet_status_snapshot(preset=preset, control_type=type)


@app.post("/api/controlnet/assets/download/start")
async def controlnet_assets_download_start(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = payload or {}
    raw_keys = payload.get("assets")
    if isinstance(raw_keys, list) and raw_keys:
        selected_keys = [str(item) for item in raw_keys if str(item) in CONTROLNET_OPTIONAL_ARTIFACTS]
    else:
        preset = str(payload.get("preset") or "").strip()
        control_type = str(payload.get("type") or payload.get("control_type") or "").strip()
        selected_keys = [item["key"] for item in _controlnet_status_snapshot(preset=preset, control_type=control_type)["missing_assets"]]
    if not selected_keys:
        raise HTTPException(status_code=400, detail="No missing ControlNet dependency assets selected.")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "kind": "controlnet_assets",
        "status": "queued",
        "progress": 0,
        "message": "Queued optional ControlNet dependency download.",
        "assets": selected_keys,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_controlnet_assets_download_job(job_id, selected_keys))
    return download_jobs[job_id]


def _nvidia_pid_profile(profile: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    key = str(profile or "lowvram_zimage_2k").strip().lower()
    base = dict(NVIDIA_PID_PROFILES.get(key) or NVIDIA_PID_PROFILES["lowvram_zimage_2k"])
    base["key"] = key if key in NVIDIA_PID_PROFILES else "lowvram_zimage_2k"
    overrides = overrides or {}
    for field in ("backbone", "checkpoint"):
        value = str(overrides.get(field) or "").strip().lower()
        if value:
            base[field] = value
    if _number_or_none(overrides.get("scale")) is not None:
        base["scale"] = max(1, min(8, int(float(overrides["scale"]))))
    if _number_or_none(overrides.get("steps")) is not None:
        base["steps"] = max(1, min(12, int(float(overrides["steps"]))))
    if _number_or_none(overrides.get("cfg")) is not None:
        base["cfg"] = max(0.1, min(5.0, float(overrides["cfg"])))
    return base


def _nvidia_pid_checkpoint_relpath(profile: dict[str, Any]) -> str:
    backbone = str(profile.get("backbone") or "zimage").lower()
    checkpoint = str(profile.get("checkpoint") or "2k").lower()
    registry_key = "flux" if backbone in {"zimage", "flux"} else backbone
    if registry_key not in {"flux", "flux2", "sd3"}:
        registry_key = "flux"
    if checkpoint not in {"2k", "2kto4k"}:
        checkpoint = "2k"
    prefix = "PiD_res2kto4k_sr4x" if checkpoint == "2kto4k" else "PiD_res2k_sr4x"
    return f"checkpoints/{prefix}_official_{registry_key}_distill_4step/model_ema_bf16.pth"


def _preferred_model_category_dir(category: str) -> Path:
    roots = settings.model_sources.get(category) or []
    for root in roots:
        try:
            path = Path(root).expanduser()
        except Exception:
            continue
        if path.exists() or path.parent.exists():
            return path
    return settings.models_dir / category


def _model_category_roots(category: str) -> list[Path]:
    roots = [settings.models_dir / category]
    roots.extend(Path(root).expanduser() for root in settings.model_sources.get(category, []))
    seen: set[str] = set()
    unique: list[Path] = []
    for root in roots:
        try:
            key = str(root.resolve())
        except Exception:
            key = str(root)
        if key in seen:
            continue
        seen.add(key)
        unique.append(root)
    return unique


def _find_existing_model_asset(category: str, relpath: str) -> Path | None:
    expected_tail = Path("nvidia_pid") / Path(relpath).parent.name / Path(relpath).name
    if category == "vae":
        expected_tail = Path("nvidia_pid") / Path(relpath).name
    for root in _model_category_roots(category):
        direct = root / expected_tail
        if direct.is_file() and direct.stat().st_size > 1024 * 1024:
            return direct
        try:
            matches = list(root.rglob(Path(relpath).name)) if root.exists() else []
        except Exception:
            matches = []
        for match in matches:
            if match.is_file() and match.stat().st_size > 1024 * 1024:
                return match
    return None


def _nvidia_pid_asset_category(relpath: str) -> str:
    name = Path(relpath).name.lower()
    if name.startswith("model_ema"):
        return "checkpoints"
    if name.endswith(".safetensors") and ("ae" in name or "vae" in relpath.lower()):
        return "vae"
    return "upscale_models"


def _nvidia_pid_assets(profile: dict[str, Any]) -> list[dict[str, Any]]:
    backbone = str(profile.get("backbone") or "zimage").lower()
    assets = [_nvidia_pid_checkpoint_relpath(profile)]
    if backbone in {"zimage", "flux"}:
        assets.append("checkpoints/ae.safetensors")
    elif backbone == "flux2":
        assets.append("checkpoints/flux2_ae.safetensors")
    elif backbone == "sd3":
        assets.append("checkpoints/sd3_vae/vae/diffusion_pytorch_model.safetensors")
    pid_dir = _nvidia_pid_dir()
    items: list[dict[str, Any]] = []
    for relpath in assets:
        catalog_path, catalog_category = _nvidia_pid_catalog_path(relpath)
        vendor_path = pid_dir / relpath
        existing = _find_existing_model_asset(catalog_category, relpath)
        exists = (
            (vendor_path.is_file() and vendor_path.stat().st_size > 1024 * 1024)
            or (catalog_path.is_file() and catalog_path.stat().st_size > 1024 * 1024)
            or existing is not None
        )
        items.append({
            "relpath": relpath,
            "path": str(vendor_path),
            "catalog_path": str(catalog_path),
            "catalog_category": catalog_category,
            "detected_path": str(existing) if existing else "",
            "size_bytes": NVIDIA_PID_ASSET_SIZES.get(relpath, 0),
            "exists": exists,
        })
    return items


def _nvidia_pid_catalog_path(relpath: str) -> tuple[Path, str]:
    name = Path(relpath).name
    if name.startswith("model_ema"):
        profile_folder = Path(relpath).parts[1] if len(Path(relpath).parts) > 1 else "pid"
        return _preferred_model_category_dir("checkpoints") / "nvidia_pid" / profile_folder / name, "checkpoints"
    if name.endswith(".safetensors") and ("ae" in name or "vae" in relpath.lower()):
        return _preferred_model_category_dir("vae") / "nvidia_pid" / name, "vae"
    return _preferred_model_category_dir("upscale_models") / "nvidia_pid" / name, "upscale_models"


def _ensure_model_alias(source: Path, target: Path) -> None:
    if not source.exists():
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return
    try:
        os.link(source, target)
        return
    except Exception:
        pass
    try:
        target.symlink_to(source)
        return
    except Exception:
        pass
    marker = target.with_suffix(target.suffix + ".nexus-link.json")
    marker.write_text(
        json.dumps({"source": str(source), "target": str(target), "kind": "model_alias"}, indent=2),
        encoding="utf-8",
    )


def _organize_nvidia_pid_assets(profile: dict[str, Any]) -> list[dict[str, Any]]:
    organized: list[dict[str, Any]] = []
    pid_dir = _nvidia_pid_dir()
    for asset in _nvidia_pid_assets(profile):
        source = pid_dir / str(asset["relpath"])
        detected = Path(str(asset.get("detected_path") or "")) if asset.get("detected_path") else None
        if (not source.exists()) and detected and detected.exists():
            _ensure_model_alias(detected, source)
        if (not source.exists()) and Path(str(asset.get("catalog_path") or "")).exists():
            _ensure_model_alias(Path(str(asset["catalog_path"])), source)
        target, category = _nvidia_pid_catalog_path(str(asset["relpath"]))
        _ensure_model_alias(source, target)
        item = dict(asset)
        item["catalog_path"] = str(target)
        item["catalog_category"] = category
        item["catalog_exists"] = target.exists()
        organized.append(item)
    ensure_model_tree(settings)
    return organized


def _ensure_nvidia_rtx_catalog_marker() -> dict[str, Any]:
    root = _preferred_model_category_dir("upscale_models") / "nvidia_rtx"
    root.mkdir(parents=True, exist_ok=True)
    marker = root / "RTXVideoSuperResolution.nexus-upscale.json"
    payload = {
        "name": "NVIDIA RTX Video Super Resolution",
        "engine": "nvidia_rtx",
        "category": "upscale_models",
        "custom_node": "Nvidia_RTX_Nodes_ComfyUI",
        "python_package": "nvvfx",
        "model_required": False,
        "notes": "Runtime engine marker; NVIDIA RTX VSR uses nvvfx and does not require a safetensors/pth checkpoint.",
    }
    if not marker.exists():
        marker.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["path"] = str(marker)
    return payload


def _nvidia_pid_prepare_status(profile: str | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    selected = _nvidia_pid_profile(profile, overrides)
    assets = _nvidia_pid_assets(selected)
    missing = [asset for asset in assets if not asset["exists"]]
    if _nvidia_pid_source_ready() and not missing:
        assets = _organize_nvidia_pid_assets(selected)
        missing = [asset for asset in assets if not asset.get("exists")]
    return {
        "profile": selected,
        "profiles": list(NVIDIA_PID_PROFILES.values()),
        "profile_keys": list(NVIDIA_PID_PROFILES.keys()),
        "pid_dir": str(_nvidia_pid_dir()),
        "source_ready": _nvidia_pid_source_ready(),
        "assets": assets,
        "missing_assets": missing,
        "prepared": _nvidia_pid_source_ready() and not missing,
        "estimated_download_bytes": sum(int(asset.get("size_bytes") or 0) for asset in missing),
    }


def _nvidia_pid_clone_source(job_id: str) -> None:
    if _nvidia_pid_source_ready():
        return
    target = _nvidia_pid_dir()
    if target.exists() and any(target.iterdir()):
        raise RuntimeError(f"PiD source directory exists but is incomplete: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    git = shutil.which("git")
    if not git:
        raise RuntimeError("git is required to download NVIDIA PiD source through the front.")
    _update_download_job(job_id, {"status": "downloading", "progress": 1, "message": "Cloning NVIDIA PiD source."})
    proc = subprocess.run([git, "clone", "--depth", "1", NVIDIA_PID_SOURCE_REPO, str(target)], cwd=settings.project_root, capture_output=True, text=True)
    if proc.returncode != 0 or not _nvidia_pid_source_ready():
        raise RuntimeError((proc.stderr or proc.stdout or "NVIDIA PiD source clone failed.").strip()[:1000])


async def _run_nvidia_pid_download_job(job_id: str, profile: str, overrides: dict[str, Any] | None = None) -> None:
    try:
        selected_status = _nvidia_pid_prepare_status(profile, overrides)
        selected = selected_status["profile"]
        _update_download_job(
            job_id,
            {
                "status": "downloading",
                "progress": 0,
                "message": f"Preparing NVIDIA PiD {selected.get('label')}.",
                "profile": selected,
                "bytes_total": selected_status["estimated_download_bytes"],
            },
        )
        await asyncio.to_thread(_nvidia_pid_clone_source, job_id)
        for asset in _nvidia_pid_assets(selected):
            relpath = str(asset["relpath"])
            target = _nvidia_pid_dir() / relpath
            detected_path = Path(str(asset.get("detected_path") or "")) if asset.get("detected_path") else None
            catalog_path = Path(str(asset.get("catalog_path") or "")) if asset.get("catalog_path") else None
            if asset["exists"]:
                if not target.exists() and detected_path and detected_path.exists():
                    _ensure_model_alias(detected_path, target)
                if not target.exists() and catalog_path and catalog_path.exists():
                    _ensure_model_alias(catalog_path, target)
                _update_download_job(job_id, {"message": f"PiD asset already exists: {Path(relpath).name}"})
                continue
            url = f"{NVIDIA_PID_HF_BASE}/{quote(relpath, safe='/')}?download=true"
            _update_download_job(job_id, {"message": f"Downloading PiD asset: {Path(relpath).name}"})
            await asyncio.to_thread(_download_url_to_file, url, target, job_id)
        organized = _organize_nvidia_pid_assets(selected)
        final_status = _nvidia_pid_prepare_status(str(selected.get("key") or profile), selected)
        _update_download_job(
            job_id,
            {
                "status": "downloaded" if final_status["prepared"] else "failed",
                "progress": 100,
                "message": f"NVIDIA PiD {selected.get('label')} ready." if final_status["prepared"] else "NVIDIA PiD prepare is incomplete.",
                "prepared": final_status["prepared"],
                "profile": selected,
                "assets": organized,
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            },
        )
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


def _pose_root() -> Path:
    return settings.output_dir / "poses"


def _pose_slug(value: Any, fallback: str = "pose") -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(value or "").strip()).strip("._-")
    return text[:80] or fallback


def _pose_data_url_bytes(value: Any) -> tuple[bytes, str] | None:
    text = str(value or "").strip()
    if not text.startswith("data:image/") or "," not in text:
        return None
    header, encoded = text.split(",", 1)
    ext = "png"
    media = header.split(";", 1)[0].split(":", 1)[-1].lower()
    if media.endswith("jpeg"):
        ext = "jpg"
    elif media.endswith("webp"):
        ext = "webp"
    elif media.endswith("gif"):
        ext = "gif"
    try:
        return base64.b64decode(encoded), ext
    except Exception:
        return None


def _pose_qwen_lora_target() -> Path:
    return settings.models_dir / "loras" / "qwen" / "VNCCS" / POSE_QWEN_LORA_ARTIFACT["filename"]


def _pose_qwen_controlnet_target() -> Path:
    return settings.models_dir / "controlnet" / "qwen" / POSE_QWEN_CONTROLNET_ARTIFACT["filename"]


def _pose_file_record(path: Path) -> dict[str, Any] | None:
    try:
        relative = path.relative_to(settings.output_dir).as_posix()
    except Exception:
        return None
    return {
        "filename": path.name,
        "path": str(path),
        "relative_path": relative,
        "url": f"/outputs/{quote(relative, safe='/')}",
        "modified": path.stat().st_mtime,
        "size_bytes": path.stat().st_size,
    }


def _pose_library_items(limit: int = 240) -> list[dict[str, Any]]:
    root = _pose_root()
    if not root.exists():
        return []
    items: list[dict[str, Any]] = []
    for metadata_path in sorted(root.rglob("pose.nexus.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            metadata = {}
        pose_dir = metadata_path.parent
        preview = next((candidate for candidate in (pose_dir / "preview.png", pose_dir / "preview.jpg", pose_dir / "openpose.png", pose_dir / "dwpose.png") if candidate.exists()), None)
        guide = next((candidate for candidate in (pose_dir / "dwpose.png", pose_dir / "openpose.png") if candidate.exists()), None)
        lighting = pose_dir / "lighting.png"
        folder = pose_dir.parent.name if pose_dir.parent != root else ""
        items.append(
            {
                "id": f"{folder}/{pose_dir.name}".strip("/"),
                "name": metadata.get("name") or pose_dir.name,
                "folder": metadata.get("folder") or folder,
                "tags": metadata.get("tags") or [],
                "prompt": metadata.get("prompt") or "",
                "compatibility": metadata.get("compatibility") or [],
                "metadata": metadata,
                "preview": _pose_file_record(preview) if preview else None,
                "guide": _pose_file_record(guide) if guide else None,
                "lighting": _pose_file_record(lighting) if lighting.exists() else None,
                "json": _pose_file_record(metadata_path),
                "modified": metadata_path.stat().st_mtime,
            }
        )
        if len(items) >= limit:
            break
    return items


def _pose_engine_status() -> dict[str, Any]:
    pose_models = settings.models_dir / "pose"
    mediapipe_model = pose_models / "mediapipe" / "pose_landmarker_full.task"
    mediapipe_installed = importlib.util.find_spec("mediapipe") is not None
    mediapipe_model_ready = mediapipe_model.exists() and mediapipe_model.stat().st_size >= 1 * 1024 * 1024
    dwpose_root = pose_models / "dwpose"
    sam_root = pose_models / "sam3d_body"
    fast_sam_runtime = settings.project_root / "runtime" / "fast-sam3d-body"
    sam_runtime = settings.project_root / "runtime" / "sam3d-body"
    qwen_target = _pose_qwen_lora_target()
    qwen_controlnet_target = _pose_qwen_controlnet_target()
    return {
        "paths": {
            "library": str(_pose_root()),
            "mediapipe": str(mediapipe_model.parent),
            "dwpose": str(dwpose_root),
            "sam3d_body": str(sam_root),
            "sam3d_runtime": str(sam_runtime),
            "fast_sam3d_runtime": str(fast_sam_runtime),
            "qwen_lora": str(qwen_target),
            "qwen_pose_controlnet": str(qwen_controlnet_target),
        },
        "engines": {
            "automatic": {
                "installed": True,
                "ready": True,
                "label": "Automatic Browser Pose",
                "note": "Uses browser MediaPipe first; heavy server engines remain optional and isolated.",
            },
            "manual": {"installed": True, "ready": True, "label": "Manual"},
            "mediapipe": {
                "installed": mediapipe_installed,
                "model_ready": mediapipe_model_ready,
                "ready": mediapipe_installed and mediapipe_model_ready,
                "label": "MediaPipe Pose",
                "note": "Uses permissive landmark capture. The browser can use the remote model; backend capture uses this local .task model.",
                "download": POSE_MEDIAPIPE_LANDMARKER_ARTIFACT,
            },
            "dwpose": {
                "installed": dwpose_root.exists() and any(dwpose_root.rglob("*")),
                "ready": False,
                "label": "DWPose / MMPose",
                "note": "Optional whole-body engine. Nexus does not install this heavy stack automatically.",
            },
            "sam3d_body": {
                "installed": sam_runtime.exists() or sam_root.exists(),
                "checkpoint_ready": any(sam_root.rglob("*.ckpt")) if sam_root.exists() else False,
                "ready": False,
                "label": "SAM 3D Body",
                "note": "Optional gated Hugging Face model; install only after user confirmation and license access.",
            },
            "fast_sam3d_body": {
                "installed": fast_sam_runtime.exists(),
                "ready": False,
                "label": "Fast SAM 3D Body",
                "note": "Optional isolated accelerator environment; not installed into Nexus/Comfy.",
            },
        },
        "qwen_lora": {
            "installed": qwen_target.exists() and qwen_target.stat().st_size > 100 * 1024 * 1024,
            "filename": POSE_QWEN_LORA_ARTIFACT["filename"],
            "version": POSE_QWEN_LORA_ARTIFACT["version"],
            "size_bytes": POSE_QWEN_LORA_ARTIFACT["size_bytes"],
            "url": POSE_QWEN_LORA_ARTIFACT["url"],
            "path": str(qwen_target),
        },
        "qwen_pose_controlnet": {
            "installed": qwen_controlnet_target.exists() and qwen_controlnet_target.stat().st_size > 100 * 1024 * 1024,
            "filename": POSE_QWEN_CONTROLNET_ARTIFACT["filename"],
            "size_bytes": POSE_QWEN_CONTROLNET_ARTIFACT["size_bytes"],
            "url": POSE_QWEN_CONTROLNET_ARTIFACT["url"],
            "path": str(qwen_controlnet_target),
            "label": POSE_QWEN_CONTROLNET_ARTIFACT["label"],
        },
    }


def _pose_capture_from_image(payload: dict[str, Any]) -> dict[str, Any]:
    image_value = str(payload.get("image") or "")
    engine = str(payload.get("engine") or "mediapipe").strip().lower()
    decoded = _pose_data_url_bytes(image_value)
    width = int(payload.get("width") or 768)
    height = int(payload.get("height") or 1024)
    if decoded:
        try:
            from PIL import Image

            with Image.open(BytesIO(decoded[0])) as image:
                width, height = image.size
        except Exception:
            pass
    known_engines = {"automatic", "manual", "mediapipe", "dwpose", "sam3d_body", "fast_sam3d_body"}
    normalized_engine = engine if engine in known_engines else "automatic"
    if normalized_engine == "manual":
        return {
            "status": "manual",
            "engine": "manual",
            "engine_used": "manual",
            "message": "Manual mode does not auto-retarget. Use Auto or MediaPipe to capture from an image.",
            "image_size": {"width": width, "height": height},
        }
    if normalized_engine in {"automatic", "mediapipe"}:
        if importlib.util.find_spec("mediapipe") is None:
            return {
                "status": "missing_dependency",
                "engine": "mediapipe",
                "engine_used": "none",
                "message": "Browser MediaPipe failed and the backend MediaPipe package is not installed. Confirm the optional dependency before server-side capture.",
                "image_size": {"width": width, "height": height},
                "dependency": {
                    "package": "mediapipe",
                    "destination": str(settings.models_dir / "pose" / "mediapipe"),
                    "optional": True,
                },
            }
        if not decoded:
            return {
                "status": "invalid_image",
                "engine": "mediapipe",
                "engine_used": "none",
                "message": "Pose capture needs an image data URL.",
                "image_size": {"width": width, "height": height},
            }
        try:
            import mediapipe as mp
            import numpy as np
            from PIL import Image

            with Image.open(BytesIO(decoded[0])) as image:
                rgb_image = image.convert("RGB")
                width, height = rgb_image.size
                frame = np.ascontiguousarray(np.asarray(rgb_image))

            landmark_list = None
            engine_used = "mediapipe-backend"
            if hasattr(mp, "solutions") and hasattr(getattr(mp, "solutions", None), "pose"):
                for complexity, confidence in ((2, 0.35), (1, 0.25), (0, 0.15)):
                    with mp.solutions.pose.Pose(
                        static_image_mode=True,
                        model_complexity=complexity,
                        enable_segmentation=False,
                        min_detection_confidence=confidence,
                    ) as pose:
                        result = pose.process(frame)
                    pose_landmarks = getattr(result, "pose_landmarks", None)
                    if pose_landmarks and getattr(pose_landmarks, "landmark", None):
                        landmark_list = pose_landmarks.landmark
                        engine_used = f"mediapipe-solutions-c{complexity}"
                        break
            else:
                model_path = settings.models_dir / "pose" / "mediapipe" / POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["filename"]
                if not model_path.exists() or model_path.stat().st_size < 1 * 1024 * 1024:
                    return {
                        "status": "missing_dependency",
                        "engine": "mediapipe",
                        "engine_used": "none",
                        "message": "MediaPipe Tasks is installed, but the local pose_landmarker_full.task model is missing. Confirm the small model download before server-side capture.",
                        "image_size": {"width": width, "height": height},
                        "dependency": {
                            "package": "mediapipe-pose-landmarker",
                            "source": POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["url"],
                            "size_bytes": POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["size_bytes"],
                            "destination": str(model_path),
                            "optional": True,
                        },
                    }
                from mediapipe.tasks.python import BaseOptions
                from mediapipe.tasks.python.vision import PoseLandmarker, PoseLandmarkerOptions, RunningMode

                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
                for confidence in (0.35, 0.25, 0.15):
                    options = PoseLandmarkerOptions(
                        base_options=BaseOptions(model_asset_path=str(model_path)),
                        running_mode=RunningMode.IMAGE,
                        num_poses=1,
                        min_pose_detection_confidence=confidence,
                        min_pose_presence_confidence=0.15,
                        min_tracking_confidence=0.15,
                        output_segmentation_masks=False,
                    )
                    with PoseLandmarker.create_from_options(options) as landmarker:
                        result = landmarker.detect(mp_image)
                    pose_landmarks = getattr(result, "pose_landmarks", None) or []
                    if pose_landmarks:
                        landmark_list = pose_landmarks[0]
                        engine_used = f"mediapipe-tasks-{confidence:.2f}"
                        break

            if not landmark_list:
                return {
                    "status": "no_pose",
                    "engine": "mediapipe",
                    "engine_used": engine_used,
                    "message": "MediaPipe did not detect a full body pose in the reference image.",
                    "image_size": {"width": width, "height": height},
                }
            names = {
                0: "nose",
                11: "left_shoulder",
                12: "right_shoulder",
                13: "left_elbow",
                14: "right_elbow",
                15: "left_wrist",
                16: "right_wrist",
                23: "left_hip",
                24: "right_hip",
                25: "left_knee",
                26: "right_knee",
                27: "left_ankle",
                28: "right_ankle",
            }
            landmarks: dict[str, list[float]] = {}
            for index, name in names.items():
                if index >= len(landmark_list):
                    continue
                item = landmark_list[index]
                landmarks[name] = [
                    float(getattr(item, "x", 0.0) or 0.0),
                    float(getattr(item, "y", 0.0) or 0.0),
                    float(getattr(item, "z", 0.0) or 0.0),
                    float(getattr(item, "visibility", 1.0) or 0.0),
                ]
            if "left_shoulder" in landmarks and "right_shoulder" in landmarks:
                left = landmarks["left_shoulder"]
                right = landmarks["right_shoulder"]
                landmarks["neck"] = [
                    (left[0] + right[0]) / 2,
                    (left[1] + right[1]) / 2,
                    (left[2] + right[2]) / 2,
                    min(left[3], right[3]),
                ]
            return {
                "status": "ready",
                "engine": "mediapipe",
                "engine_used": engine_used,
                "message": "MediaPipe backend landmarks retargeted to the mannequin.",
                "image_size": {"width": width, "height": height},
                "landmarks": landmarks,
            }
        except Exception as exc:
            return {
                "status": "capture_failed",
                "engine": "mediapipe",
                "engine_used": "mediapipe-backend",
                "message": f"MediaPipe backend capture failed: {exc}",
                "image_size": {"width": width, "height": height},
            }
    if normalized_engine in {"dwpose", "sam3d_body", "fast_sam3d_body"}:
        return {
            "status": "dependency_required",
            "engine": normalized_engine,
            "engine_used": "none",
            "message": f"{normalized_engine} is optional and must be installed in an isolated runtime before capture.",
            "image_size": {"width": width, "height": height},
        }
    return {
        "status": "missing_dependency",
        "engine": normalized_engine,
        "engine_used": "none",
        "message": "Server-side pose capture is not configured. Use browser Auto capture or install an optional pose engine.",
        "image_size": {"width": width, "height": height},
    }


async def _run_pose_qwen_lora_download_job(job_id: str) -> None:
    try:
        target = _pose_qwen_lora_target()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 100 * 1024 * 1024:
            result = {
                "status": "downloaded",
                "already_downloaded": True,
                "filename": target.name,
                "path": str(target),
                "relative_path": _download_relative_path(target),
                "bytes_downloaded": target.stat().st_size,
                "bytes_total": target.stat().st_size,
                "progress": 100,
            }
        else:
            _update_download_job(job_id, {"status": "downloading", "progress": 0, "message": f"Downloading {POSE_QWEN_LORA_ARTIFACT['label']}"})
            result = await asyncio.to_thread(_download_url_to_file, POSE_QWEN_LORA_ARTIFACT["url"], target, job_id)
        ensure_model_tree(settings)
        _update_download_job(job_id, {**result, "message": f"{POSE_QWEN_LORA_ARTIFACT['label']} ready.", "completed_at": datetime.now().isoformat(timespec="seconds")})
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/pose/status")
async def pose_status() -> dict[str, Any]:
    status = _pose_engine_status()
    status["library_count"] = len(_pose_library_items())
    return status


@app.get("/api/pose/library")
async def pose_library() -> list[dict[str, Any]]:
    return _pose_library_items()


@app.post("/api/pose/save")
async def pose_save(payload: dict[str, Any]) -> dict[str, Any]:
    folder = _pose_slug(payload.get("folder"), "library")
    name = _pose_slug(payload.get("name"), f"pose_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    target_dir = (_pose_root() / folder / name).resolve()
    if not target_dir.is_relative_to(_pose_root().resolve()):
        raise HTTPException(status_code=400, detail="Invalid pose path.")
    target_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, str] = {}
    for field, filename in {
        "preview_image": "preview",
        "openpose_image": "openpose",
        "dwpose_image": "dwpose",
        "lighting_image": "lighting",
    }.items():
        decoded = _pose_data_url_bytes(payload.get(field))
        if not decoded:
            continue
        data, ext = decoded
        path = target_dir / f"{filename}.{ext}"
        path.write_bytes(data)
        written[field] = str(path)

    metadata = {
        "name": name,
        "folder": folder,
        "tags": payload.get("tags") or [],
        "prompt": payload.get("prompt") or "",
        "compatibility": payload.get("compatibility") or ["Qwen", "Flux", "SDXL", "LTX"],
        "engine": payload.get("engine") or "manual",
        "pose": payload.get("pose") or {},
        "camera": payload.get("camera") or {},
        "lights": payload.get("lights") or {},
        "guide": payload.get("guide") or {},
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "files": written,
    }
    metadata_path = target_dir / "pose.nexus.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"saved": True, "item": _pose_library_items(limit=1)[0] if _pose_library_items(limit=1) else {"metadata": metadata, "json": _pose_file_record(metadata_path)}}


@app.post("/api/pose/capture")
async def pose_capture(payload: dict[str, Any]) -> dict[str, Any]:
    return _pose_capture_from_image(payload)


@app.post("/api/pose/dependencies/download")
async def pose_dependencies_download(payload: dict[str, Any]) -> dict[str, Any]:
    engine = str(payload.get("engine") or "").strip().lower()
    if engine in {"sam3d_body", "fast_sam3d_body", "dwpose", "mmpose"}:
        return {
            "status": "manual_required",
            "engine": engine,
            "message": "This engine is optional and heavy. Nexus keeps it isolated; install will be added behind a confirmed UI flow before any dependency is downloaded.",
            "status_snapshot": _pose_engine_status(),
        }
    if engine == "mediapipe":
        target = settings.models_dir / "pose" / "mediapipe" / POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["filename"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size >= 1 * 1024 * 1024:
            return {
                "status": "ready",
                "engine": engine,
                "message": "MediaPipe Pose Landmarker model is already ready.",
                "path": str(target),
                "status_snapshot": _pose_engine_status(),
            }
        job_id = uuid.uuid4().hex[:12]
        download_jobs[job_id] = {
            "job_id": job_id,
            "status": "downloading",
            "progress": 0,
            "message": f"Downloading {POSE_MEDIAPIPE_LANDMARKER_ARTIFACT['label']}.",
            "filename": POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["filename"],
            "url": POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["url"],
            "size_bytes": POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["size_bytes"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        result = await asyncio.to_thread(_download_url_to_file, POSE_MEDIAPIPE_LANDMARKER_ARTIFACT["url"], target, job_id)
        return {
            **result,
            "status": "ready",
            "engine": engine,
            "message": "MediaPipe Pose Landmarker model is ready.",
            "status_snapshot": _pose_engine_status(),
        }
    if engine == "qwen_pose_controlnet":
        target = _pose_qwen_controlnet_target()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and target.stat().st_size > 100 * 1024 * 1024:
            return {
                "status": "ready",
                "engine": engine,
                "message": "Qwen POSE ControlNet Union is already ready.",
                "path": str(target),
                "status_snapshot": _pose_engine_status(),
            }
        job_id = uuid.uuid4().hex[:12]
        download_jobs[job_id] = {
            "job_id": job_id,
            "status": "downloading",
            "progress": 0,
            "message": f"Downloading {POSE_QWEN_CONTROLNET_ARTIFACT['label']}.",
            "filename": POSE_QWEN_CONTROLNET_ARTIFACT["filename"],
            "url": POSE_QWEN_CONTROLNET_ARTIFACT["url"],
            "size_bytes": POSE_QWEN_CONTROLNET_ARTIFACT["size_bytes"],
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        result = await asyncio.to_thread(_download_url_to_file, POSE_QWEN_CONTROLNET_ARTIFACT["url"], target, job_id)
        ensure_model_tree(settings)
        return {
            **result,
            "status": "ready",
            "engine": engine,
            "message": "Qwen POSE ControlNet Union is ready. Restart UI + backend if the model selector does not show it immediately.",
            "status_snapshot": _pose_engine_status(),
        }
    raise HTTPException(status_code=404, detail=f"Unknown POSE dependency engine: {engine or 'empty'}")


@app.post("/api/pose/qwen-lora/download")
async def pose_qwen_lora_download() -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": f"Queued {POSE_QWEN_LORA_ARTIFACT['label']} download.",
        "filename": POSE_QWEN_LORA_ARTIFACT["filename"],
        "url": POSE_QWEN_LORA_ARTIFACT["url"],
        "size_bytes": POSE_QWEN_LORA_ARTIFACT["size_bytes"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_pose_qwen_lora_download_job(job_id))
    return download_jobs[job_id]


def _ltx_hf_lora_artifact(kind: str) -> dict[str, str]:
    normalized = kind.strip().lower()
    if normalized == "ltx_detailer":
        normalized = "detailer"
    artifact = LTX_HF_LORA_ARTIFACTS.get(normalized)
    if not artifact:
        raise HTTPException(status_code=404, detail=f"Unknown LTX artifact: {kind}")
    return artifact


def _ltx_hf_lora_target(kind: str) -> Path:
    artifact = _ltx_hf_lora_artifact(kind)
    normalized = kind.strip().lower()
    if normalized == "ltx_detailer":
        normalized = "detailer"
    if normalized == "denoise":
        return settings.models_dir / "denoise_models" / artifact["filename"]
    if normalized in {"flashvsr", "seedvr2"}:
        folder = "FlashVSR" if normalized == "flashvsr" else "SeedVR2"
        return settings.models_dir / "video_restore_models" / folder / artifact["filename"]
    if normalized == "face_restore":
        return settings.models_dir / "face_restore_models" / artifact["filename"]
    if normalized == "transition":
        return settings.models_dir / "loras" / "ltx_transition" / artifact["filename"]
    return settings.models_dir / "loras" / "ltx_ic" / artifact["filename"]


def _ltx_hf_lora_installed_path(kind: str) -> Path:
    artifact = _ltx_hf_lora_artifact(kind)
    filename = artifact["filename"]
    normalized = kind.strip().lower()
    if normalized == "ltx_detailer":
        normalized = "detailer"
    if normalized == "denoise":
        candidates = (
            settings.models_dir / "denoise_models" / filename,
            settings.models_dir / "upscale_models" / filename,
        )
    elif normalized == "flashvsr":
        candidates = (
            settings.models_dir / "FlashVSR" / filename,
            settings.models_dir / "video_restore_models" / "FlashVSR" / filename,
            settings.models_dir / "video_restore_models" / filename,
        )
    elif normalized == "seedvr2":
        candidates = (
            settings.models_dir / "SEEDVR2" / filename,
            settings.models_dir / "video_restore_models" / "SeedVR2" / filename,
            settings.models_dir / "video_restore_models" / filename,
            settings.models_dir / "SeedVR2" / filename,
        )
    elif normalized == "face_restore":
        candidates = (
            settings.models_dir / "face_restore_models" / filename,
            settings.models_dir / "facerestore_models" / filename,
            settings.models_dir / "GFPGAN" / filename,
        )
    elif normalized == "transition":
        candidates = (
            settings.models_dir / "loras" / "ltx_transition" / filename,
            settings.models_dir / "loras" / "ltx" / filename,
            settings.models_dir / "loras" / filename,
        )
    else:
        candidates = (
            settings.models_dir / "loras" / "ltx_ic" / filename,
            settings.models_dir / "loras" / "ltx" / filename,
            settings.models_dir / "loras" / filename,
        )
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
            return candidate
    return candidates[0]


async def _run_ltx_hf_lora_download_job(job_id: str, kind: str) -> None:
    try:
        artifact = _ltx_hf_lora_artifact(kind)
        target = _ltx_hf_lora_installed_path(kind)
        if target.exists() and target.stat().st_size > 1024 * 1024:
            result = {
                "status": "downloaded",
                "already_downloaded": True,
                "filename": target.name,
                "path": str(target),
                "relative_path": _download_relative_path(target),
                "bytes_downloaded": target.stat().st_size,
                "bytes_total": target.stat().st_size,
                "progress": 100,
            }
        else:
            _update_download_job(job_id, {"status": "downloading", "progress": 0, "message": f"Downloading {artifact['label']}"})
            result = await asyncio.to_thread(_download_url_to_file, artifact["url"], target, job_id)
        ensure_model_tree(settings)
        _update_download_job(job_id, {**result, "message": f"{artifact['label']} ready.", "completed_at": datetime.now().isoformat(timespec="seconds")})
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/ltx/detailer/status")
async def ltx_detailer_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("detailer")
    target = _ltx_hf_lora_installed_path("detailer")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir / "loras")).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/ltx/detailer/download/start")
async def ltx_detailer_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("detailer")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued LTX IC-LoRA Detailer download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "detailer"))
    return download_jobs[job_id]


@app.get("/api/ltx/control/status")
async def ltx_control_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("control")
    target = _ltx_hf_lora_installed_path("control")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir / "loras")).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/ltx/control/download/start")
async def ltx_control_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("control")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued LTX 2.3 IC-LoRA Union Control download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "control"))
    return download_jobs[job_id]


@app.get("/api/ltx/transition/status")
async def ltx_transition_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("transition")
    target = _ltx_hf_lora_installed_path("transition")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir / "loras")).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/ltx/transition/download/start")
async def ltx_transition_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("transition")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued LTX 2.3 Transition LoRA download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "transition"))
    return download_jobs[job_id]


@app.get("/api/ltx/cameraman/status")
async def ltx_cameraman_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("cameraman")
    target = _ltx_hf_lora_installed_path("cameraman")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir / "loras")).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/ltx/cameraman/download/start")
async def ltx_cameraman_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("cameraman")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued LTX 2.3 IC-LoRA Cameraman download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "cameraman"))
    return download_jobs[job_id]


@app.get("/api/ltx/outpaint/status")
async def ltx_outpaint_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("outpaint")
    target = _ltx_hf_lora_installed_path("outpaint")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir / "loras")).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/ltx/outpaint/download/start")
async def ltx_outpaint_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("outpaint")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued LTX 2.3 IC-LoRA Outpaint download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "outpaint"))
    return download_jobs[job_id]


def _qwen_multiangle_lora_target() -> Path:
    return settings.models_dir / "loras" / "qwen" / QWEN_MULTIANGLE_LORA_ARTIFACT["filename"]


def _qwen_multiangle_lora_installed_path() -> Path:
    filename = QWEN_MULTIANGLE_LORA_ARTIFACT["filename"]
    candidates = (
        settings.models_dir / "loras" / "qwen" / filename,
        settings.models_dir / "loras" / filename,
    )
    for candidate in candidates:
        if candidate.exists() and candidate.stat().st_size > 1024 * 1024:
            return candidate
    return candidates[0]


async def _run_qwen_multiangle_lora_download_job(job_id: str) -> None:
    try:
        artifact = QWEN_MULTIANGLE_LORA_ARTIFACT
        target = _qwen_multiangle_lora_installed_path()
        if target.exists() and target.stat().st_size > 1024 * 1024:
            result = {
                "status": "downloaded",
                "already_downloaded": True,
                "filename": target.name,
                "path": str(target),
                "relative_path": _download_relative_path(target),
                "bytes_downloaded": target.stat().st_size,
                "bytes_total": target.stat().st_size,
                "progress": 100,
            }
        else:
            target = _qwen_multiangle_lora_target()
            _update_download_job(job_id, {"status": "downloading", "progress": 0, "message": f"Downloading {artifact['label']}"})
            result = await asyncio.to_thread(_download_url_to_file, artifact["url"], target, job_id)
        ensure_model_tree(settings)
        _update_download_job(job_id, {**result, "message": f"{artifact['label']} ready.", "completed_at": datetime.now().isoformat(timespec="seconds")})
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.get("/api/qwen/multiview/status")
async def qwen_multiview_status() -> dict[str, Any]:
    artifact = QWEN_MULTIANGLE_LORA_ARTIFACT
    target = _qwen_multiangle_lora_installed_path()
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir / "loras")).replace("/", "\\"),
        "filename": artifact["filename"],
        "label": artifact["label"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
        "size_bytes": int(artifact["size_bytes"]),
    }


@app.post("/api/qwen/multiview/download/start")
async def qwen_multiview_download_start() -> dict[str, Any]:
    artifact = QWEN_MULTIANGLE_LORA_ARTIFACT
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued Qwen MultiView LoRA download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "bytes_total": int(artifact["size_bytes"]),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_qwen_multiangle_lora_download_job(job_id))
    return download_jobs[job_id]


@app.get("/api/extras/denoise/status")
async def extras_denoise_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("denoise")
    target = _ltx_hf_lora_installed_path("denoise")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir)).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/extras/denoise/download/start")
async def extras_denoise_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("denoise")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued FastDVDnet video denoise download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "denoise"))
    return download_jobs[job_id]


@app.get("/api/extras/video-restore/{engine}/status")
async def extras_video_restore_status(engine: str) -> dict[str, Any]:
    normalized = engine.strip().lower()
    if normalized not in {"flashvsr", "seedvr2", "ltx_detailer"}:
        raise HTTPException(status_code=404, detail=f"Unknown video restore engine: {engine}")
    artifact = _ltx_hf_lora_artifact(normalized)
    target = _ltx_hf_lora_installed_path(normalized)
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    expected_nodes = EXTRAS_VIDEO_RESTORE_NODES[normalized]
    node_ready = any((settings.custom_nodes_dir / name).exists() for name in expected_nodes)
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir)).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
        "engine": normalized,
        "node_ready": node_ready,
        "expected_nodes": list(expected_nodes),
    }


@app.post("/api/extras/video-restore/{engine}/download/start")
async def extras_video_restore_download_start(engine: str) -> dict[str, Any]:
    normalized = engine.strip().lower()
    if normalized not in {"flashvsr", "seedvr2", "ltx_detailer"}:
        raise HTTPException(status_code=404, detail=f"Unknown video restore engine: {engine}")
    artifact = _ltx_hf_lora_artifact(normalized)
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": f"Queued {artifact['label']} download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, normalized))
    return download_jobs[job_id]


@app.get("/api/extras/nvidia/{engine}/status")
async def extras_nvidia_status(
    engine: str,
    profile: str | None = Query(None),
    backbone: str | None = Query(None),
    checkpoint: str | None = Query(None),
    scale: int | None = Query(None),
    steps: int | None = Query(None),
    cfg: float | None = Query(None),
) -> dict[str, Any]:
    status = _nvidia_extras_status(engine)
    if engine.strip().lower() == "nvidia_pid":
        overrides = {"backbone": backbone, "checkpoint": checkpoint, "scale": scale, "steps": steps, "cfg": cfg}
        status["pid"] = _nvidia_pid_prepare_status(profile, overrides)
    return status


@app.post("/api/extras/nvidia/{engine}/download/start")
async def extras_nvidia_download_start(
    engine: str,
    profile: str = Query("lowvram_zimage_2k"),
    backbone: str | None = Query(None),
    checkpoint: str | None = Query(None),
    scale: int | None = Query(None),
    steps: int | None = Query(None),
    cfg: float | None = Query(None),
) -> dict[str, Any]:
    normalized = engine.strip().lower()
    if normalized != "nvidia_pid":
        raise HTTPException(status_code=400, detail="Only NVIDIA PiD requires a downloadable model prepare step.")
    overrides = {"backbone": backbone, "checkpoint": checkpoint, "scale": scale, "steps": steps, "cfg": cfg}
    selected = _nvidia_pid_profile(profile, overrides)
    job_id = f"download-nvidia-pid-{uuid.uuid4().hex[:10]}"
    download_jobs[job_id] = {
        "job_id": job_id,
        "kind": "nvidia_pid",
        "status": "queued",
        "progress": 0,
        "message": f"Queued NVIDIA PiD {selected.get('label')} prepare.",
        "profile": selected,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_nvidia_pid_download_job(job_id, profile, overrides))
    return download_jobs[job_id]


@app.get("/api/extras/face-restore/status")
async def extras_face_restore_status() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("face_restore")
    target = _ltx_hf_lora_installed_path("face_restore")
    installed = target.exists() and target.stat().st_size > 1024 * 1024
    return {
        "installed": installed,
        "name": str(target.relative_to(settings.models_dir)).replace("/", "\\"),
        "filename": artifact["filename"],
        "path": str(target) if installed else "",
        "url": artifact["url"],
    }


@app.post("/api/extras/face-restore/download/start")
async def extras_face_restore_download_start() -> dict[str, Any]:
    artifact = _ltx_hf_lora_artifact("face_restore")
    job_id = uuid.uuid4().hex[:12]
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued GFPGAN face restoration download.",
        "filename": artifact["filename"],
        "url": artifact["url"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    asyncio.create_task(_run_ltx_hf_lora_download_job(job_id, "face_restore"))
    return download_jobs[job_id]


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
            settings=settings,
            query=request.query,
            tag=request.tag,
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


async def _run_trellis2_download_job(job_id: str) -> None:
    target_dir = _trellis2_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        token = _huggingface_token()
        _update_download_job(job_id, {
            "status": "downloading",
            "progress": 5,
            "message": "Downloading TRELLIS.2-4B snapshot from Hugging Face.",
            "token_configured": bool(token),
        })

        def _snapshot_download() -> None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is required to download TRELLIS.2-4B.") from exc
            snapshot_download(
                repo_id=TRELLIS2_REPO_ID,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=token,
            )

        await asyncio.to_thread(_snapshot_download)
        ensure_model_tree(settings)
        _update_download_job(job_id, {
            "status": "downloaded",
            "progress": 100,
            "message": "TRELLIS.2-4B ready.",
            "path": str(target_dir),
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


async def _run_dinov3_download_job(job_id: str) -> None:
    target_dir = _dinov3_model_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    try:
        token = _huggingface_token()
        _update_download_job(job_id, {
            "status": "downloading",
            "progress": 5,
            "message": "Downloading DINOv3 ViT-L/16 snapshot from Hugging Face.",
            "token_configured": bool(token),
        })

        def _snapshot_download() -> None:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError("huggingface_hub is required to download DINOv3.") from exc
            snapshot_download(
                repo_id=DINOV3_REPO_ID,
                local_dir=str(target_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
                token=token,
                allow_patterns=[
                    "config.json",
                    "model.safetensors",
                    "preprocessor_config.json",
                    "*.json",
                    "*.txt",
                    "*.md",
                ],
            )

        try:
            await asyncio.to_thread(_snapshot_download)
        except Exception as hf_exc:
            _update_download_job(job_id, {
                "status": "downloading",
                "progress": 35,
                "message": f"Hugging Face DINOv3 download failed ({type(hf_exc).__name__}); trying Kaggle mirror.",
                "hf_error": str(hf_exc),
                "kaggle_handle": DINOV3_KAGGLE_HANDLE,
            })

            def _kaggle_download() -> None:
                try:
                    import kagglehub
                except ImportError as exc:
                    raise RuntimeError("kagglehub is required to download DINOv3 from Kaggle.") from exc
                source = Path(kagglehub.model_download(DINOV3_KAGGLE_HANDLE))
                _copy_dinov3_snapshot(source, target_dir)

            await asyncio.to_thread(_kaggle_download)
        existing = _dinov3_snapshot_files(target_dir)
        if not existing:
            raise RuntimeError("DINOv3 download finished, but model.safetensors was not found.")
        _update_download_job(job_id, {
            "status": "downloaded",
            "progress": 100,
            "message": "DINOv3 ViT-L/16 ready.",
            "path": str(target_dir),
            "existing": existing,
            "completed_at": datetime.now().isoformat(timespec="seconds"),
        })
    except Exception as exc:
        _update_download_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)})


@app.post("/api/model3d/trellis2/download/start")
async def model3d_trellis2_download_start() -> dict[str, Any]:
    status = await model3d_trellis2_status()
    job_id = uuid.uuid4().hex
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued TRELLIS.2-4B download.",
        "filename": "TRELLIS.2-4B",
        "url": f"https://huggingface.co/{TRELLIS2_REPO_ID}",
        "path": status["path"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if status["installed"]:
        download_jobs[job_id].update({"status": "downloaded", "progress": 100, "message": "TRELLIS.2-4B already installed."})
        return download_jobs[job_id]
    asyncio.create_task(_run_trellis2_download_job(job_id))
    return download_jobs[job_id]


@app.post("/api/model3d/dinov3/download/start")
async def model3d_dinov3_download_start() -> dict[str, Any]:
    status = await model3d_dinov3_status()
    job_id = uuid.uuid4().hex
    download_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Queued DINOv3 ViT-L/16 download.",
        "filename": "dinov3-vitl16-pretrain-lvd1689m",
        "url": f"https://huggingface.co/{DINOV3_REPO_ID}",
        "path": status["path"],
        "token_file": str(HF_TOKEN_PATH),
        "token_configured": status["token_configured"],
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    if status["installed"]:
        download_jobs[job_id].update({"status": "downloaded", "progress": 100, "message": "DINOv3 ViT-L/16 already installed."})
        return download_jobs[job_id]
    asyncio.create_task(_run_dinov3_download_job(job_id))
    return download_jobs[job_id]


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
    global last_generation_model_signature
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
        _normalize_ltx_outpaint_workflow_scope(request)
        assets = resolve_generation_assets(settings, request)
        if request.preset.lower() == "ltx" and float(request.cfg or 0) == 7.0:
            request.cfg = 1.0
        _ensure_ltx_default_distilled_loras(request, assets)
        _ensure_wan_4step_loras(request, assets)
        _ensure_qwen_edit_lightning_lora(request, assets)
        _ensure_qwen_multiangle_lora(request, assets)
        if request.preset.lower() == "model3d":
            requested_model = str((request.model3d or {}).get("model") or request.model_name or "microsoft/TRELLIS.2-4B")
            model3d_preflight = await _model3d_preflight_report(requested_model=requested_model)
            if model3d_preflight.get("blocking"):
                detail = " ".join(str(item) for item in model3d_preflight.get("blocking", []))
                print(f"NEXUS BTA WARN Model 3D preflight blocked before generation: {detail}", flush=True)
                raise ValueError(f"Model 3D is not ready: {detail}")
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
        if request.preset.lower() in {"ideogram4", "ideogram"}:
            missing_ideogram_assets: list[str] = []
            if not assets.get("primary_model"):
                missing_ideogram_assets.append("ideogram4_fp8_scaled.safetensors")
            if not assets.get("ideogram4_unconditional_model"):
                missing_ideogram_assets.append("ideogram4_unconditional_fp8_scaled.safetensors")
            if Path(str(assets.get("text_encoder") or "")).name.lower() != "qwen3vl_8b_fp8_scaled.safetensors":
                missing_ideogram_assets.append("qwen3vl_8b_fp8_scaled.safetensors")
            if Path(str(assets.get("vae") or "")).name.lower() != "flux2-vae.safetensors":
                missing_ideogram_assets.append("flux2-vae.safetensors")
            if missing_ideogram_assets:
                raise ValueError("Ideogram 4 missing required assets: " + ", ".join(missing_ideogram_assets) + ".")
            img2img_mode = str(getattr(request.img2img, "mode", "") or "").strip().lower()
            if "inpaint" in img2img_mode:
                raise ValueError(
                    "Ideogram 4 local Comfy route does not support true mask inpaint yet. "
                    "Use Linear Viewer ADD boxes as regional JSON guides."
                )
        _apply_inpaint_intent_prompt(request)
        reference_image_names = _prepare_reference_images(request)
        ltx_director_frame_guides: list[dict[str, Any]] = []
        ltx_director_segment_render_mode = _ltx_director_segment_render_requested(request)
        if request.preset.lower() == "ltx" and getattr(request, "workspace", "") == "director" and not ltx_director_segment_render_mode:
            ltx_director_frame_guides = [] if request.workflow_override else _prepare_ltx_director_frame_guides(request)
            if ltx_director_frame_guides:
                reference_image_names = [str(guide["image"]) for guide in ltx_director_frame_guides]
                request.activity = "img2img"
                request.workflow_override = None
                request.workflow_id = None
        ltx_director_motion_video = None if (
            request.preset.lower() == "ltx"
            and getattr(request, "workspace", "") == "director"
            and (request.workflow_override or ltx_director_segment_render_mode)
        ) else _prepare_ltx_director_motion_transfer(request)
        if ltx_director_motion_video:
            request.activity = "img2img"
            request.workflow_override = None
            request.workflow_id = None
        base_video_name = ltx_director_motion_video or _prepare_base_video(request)
        if (
            base_video_name
            and request.preset.lower() == "ltx"
            and isinstance(request.video, dict)
            and request.video.get("motion_transfer_enabled")
            and not ltx_director_segment_render_mode
        ):
            video_options = request.video or {}
            fps = max(1.0, float(_number_or_none(video_options.get("fps")) or 24))
            seconds = float(_number_or_none(video_options.get("seconds") or video_options.get("duration")) or 0)
            frames_value = _number_or_none(video_options.get("frames") or video_options.get("length"))
            frames = int(round(frames_value)) if frames_value is not None else 0
            if frames <= 0:
                if seconds <= 0:
                    seconds = max(0.25, _ffprobe_duration(settings.input_dir / base_video_name))
                frames = max(1, int(round(seconds * fps)))
            if seconds <= 0:
                seconds = max(0.25, frames / fps)
            base_video_name = _normalize_director_motion_reference(
                base_video_name,
                seconds,
                fps,
                max(64, int(request.width or 512)),
                max(64, int(request.height or 512)),
                frames,
            )
            request.img2img.base_video = str(settings.input_dir / base_video_name)
        ltx_loop_cycle = _truthy((request.video or {}).get("ltx_loop_cycle")) if request.preset.lower() == "ltx" else False
        ltx_loop_source = str((request.video or {}).get("ltx_loop_source") or "").strip().lower()
        if ltx_loop_cycle and isinstance(request.video, dict):
            request.video["motion_transfer_enabled"] = False
            request.video["motion_transfer_mode"] = "off"
            request.video["motion_transfer_control_mode"] = "off"
            base_video_name = None
            request.img2img.base_video = None
        if not ltx_loop_cycle and not base_video_name and request.preset.lower() == "ltx" and len(reference_image_names) >= 2:
            base_video_name = _prepare_ltx_motion_scaffold(reference_image_names, request)
        reference_image_name = reference_image_names[0] if reference_image_names else None
        reference_end_image_name = reference_image_names[1] if len(reference_image_names) > 1 else None
        if (
            ltx_loop_cycle
            and reference_image_name
            and not base_video_name
            and not reference_end_image_name
        ):
            reference_end_image_name = reference_image_name
        mask_image_name = _prepare_mask_image(request)
        composite_mask_image_name = _prepare_composite_mask_image(request)
        if composite_mask_image_name:
            request.img2img.composite_mask_image = composite_mask_image_name
        controlnet_image_name = _prepare_controlnet_image(request)
        if reference_image_name:
            assets["reference_image"] = reference_image_name
        if base_video_name:
            assets["base_video"] = base_video_name
            if request.preset.lower() == "ltx" and request.workflow_id == "ltx23-video-outpainting":
                outpaint_reference_image = _extract_video_first_frame(base_video_name, "nexus_ltx_outpaint_frame")
                if outpaint_reference_image:
                    assets["outpaint_reference_image"] = outpaint_reference_image
        if reference_image_names:
            assets["reference_images"] = reference_image_names
        if mask_image_name:
            assets["mask_image"] = mask_image_name
        if composite_mask_image_name:
            assets["composite_mask_image"] = composite_mask_image_name
        if controlnet_image_name:
            assets["controlnet_image"] = controlnet_image_name
        if assets.get("primary_model") and not request.model_name:
            request.model_name = assets["primary_model"]
        model_signature = (
            request.preset.lower(),
            str(assets.get("primary_model") or request.model_name or request.model_path or request.template or ""),
        )
        if last_generation_model_signature and model_signature != last_generation_model_signature and await comfy.is_running():
            if job_id:
                _update_generation_job(job_id, {"status": "preparing", "progress": 6, "message": "Clearing previous model from VRAM"}, force=True)
            await comfy.free_memory(unload_models=True, free_memory=True)
        last_generation_model_signature = model_signature
        workflow_path = workflow_registry.find(request.workflow_id, request.preset)
        if request.preset.lower() == "ltx" and not request.workflow_id:
            workflow_path = None
        if request.preset.lower() == "wan" and not request.workflow_id:
            workflow_path = None
        if base_video_name and not request.workflow_id and request.preset.lower() in {"wan", "ltx"}:
            workflow_path = None
        if request.preset.lower() == "qwen" and request.activity == "img2img" and reference_image_name:
            request.workflow_override = None
            workflow_path = None

        if job_id:
            _update_generation_job(job_id, {"status": "starting", "progress": 6, "message": "Starting embedded ComfyUI"}, force=True)
        lanpaint_installed = _ensure_lanpaint_custom_node(request)
        if lanpaint_installed and await comfy.is_running():
            await comfy.restart()
        await comfy.ensure_running()
        if job_id:
            _update_generation_job(job_id, {"status": "preparing", "progress": 7, "message": "Reading Comfy object registry"})
        _raise_if_generation_cancelled(job_id)
        object_info = await comfy.object_info()
        _raise_if_generation_cancelled(job_id)
        if request.preset.lower() == "model3d":
            node_status = _model3d_node_status(object_info)
            if node_status.get("missing"):
                missing = ", ".join(node_status["missing"][:8])
                print(f"NEXUS BTA WARN Model 3D required custom nodes missing: {missing}", flush=True)
                raise ValueError(f"Model 3D required custom nodes are missing: {missing}. Install Model 3D workflow requirements or open update.bat.")
        if request.preset.lower() in {"ideogram4", "ideogram"}:
            missing_ideogram_core = _ideogram4_missing_core_support(object_info)
            if missing_ideogram_core:
                missing = ", ".join(missing_ideogram_core[:8])
                raise ValueError(
                    "Ideogram 4 requires a newer ComfyUI core with official Day-0 Ideogram nodes. "
                    f"Missing runtime support: {missing}. Update the embedded ComfyUI runtime, then restart Nexus."
                )
        director_segment_response = await _run_ltx_director_segment_render(request, assets, object_info, job_id=job_id)
        if director_segment_response:
            _cleanup_generation_temp()
            await _release_comfy_memory_if_idle()
            _schedule_comfy_idle_release()
            return director_segment_response

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
                motion_transfer_requested = bool(base_video_name and _truthy((request.video or {}).get("motion_transfer_enabled")))
                if base_video_name and not motion_transfer_requested and not _available_comfy_node(object_info, "VHS_LoadVideo"):
                    raise ValueError("LTX video2video requires comfyui-videohelpersuite (VHS_LoadVideo).")
                motion_control_mode = str((request.video or {}).get("motion_transfer_control_mode") or "pose").strip().lower()
                ltx_motion_ic_lora_name = assets.get("cameraman_lora") if motion_control_mode == "camera" else assets.get("ic_lora")
                if motion_transfer_requested:
                    if not ltx_motion_ic_lora_name:
                        if motion_control_mode == "camera":
                            raise ValueError("LTX Motion Transfer camera mode requires LTX 2.3 IC-LoRA Cameraman under models/loras/ltx.")
                        raise ValueError("LTX Motion Transfer requires an LTX 2.3 IC-LoRA Union Control model under models/loras.")
                    motion_required_nodes = [
                        "LTXICLoRALoaderModelOnly",
                        "LTXAddVideoICLoRAGuide",
                        "LTXVImgToVideoConditionOnly",
                        "LTXVEmptyLatentAudio",
                        "LTXVConcatAVLatent",
                        "LTXVSeparateAVLatent",
                        "LTXVCropGuides",
                        "ImageResizeKJv2",
                    ]
                    if reference_end_image_name and motion_control_mode == "camera":
                        motion_required_nodes.append("LTXVAddGuideMulti")
                    if motion_control_mode == "camera":
                        motion_required_nodes.append("VHS_LoadVideo")
                    else:
                        motion_required_nodes.extend(["LoadVideo", "GetVideoComponents", "ResizeImageMaskNode", "SimpleMath+"])
                    missing_motion_nodes = [
                        node
                        for node in motion_required_nodes
                        if not _available_comfy_node(object_info, node)
                    ]
                    if motion_control_mode == "pose" and not (
                        _available_comfy_node(object_info, "DWPreprocessor")
                        or _available_comfy_node(object_info, "OpenposePreprocessor")
                    ):
                        missing_motion_nodes.append("DWPreprocessor or OpenposePreprocessor")
                    if motion_control_mode == "canny" and not (
                        _available_comfy_node(object_info, "CannyEdgePreprocessor")
                        or _available_comfy_node(object_info, "Canny")
                    ):
                        missing_motion_nodes.append("CannyEdgePreprocessor or Canny")
                    if motion_control_mode == "depth":
                        missing_motion_nodes.extend(
                            node
                            for node in (
                                "LoadVideoDepthAnythingModel",
                                "VideoDepthAnythingProcess",
                                "VideoDepthAnythingOutput",
                            )
                            if not _available_comfy_node(object_info, node)
                        )
                    if missing_motion_nodes:
                        raise ValueError("LTX Motion Transfer requires ComfyUI-LTXVideo IC-LoRA nodes: " + ", ".join(missing_motion_nodes) + ".")
                elif base_video_name and not _available_comfy_node(object_info, "LTXVAddGuide"):
                    raise ValueError("LTX video2video requires the native LTXVAddGuide node.")
                prompt = build_basic_ltx_img2video_workflow(
                    request,
                    checkpoint_name,
                    text_encoder_name,
                    reference_image_name,
                    reference_end_image_name=reference_end_image_name,
                    base_video_name=base_video_name,
                    ic_lora_name=ltx_motion_ic_lora_name,
                    text_projection_name=assets.get("text_projection"),
                    audio_vae_name=assets.get("audio_vae"),
                    video_vae_name=assets.get("video_vae") or assets.get("vae"),
                    latent_upscale_name=assets.get("latent_upscale"),
                    transition_lora_name=assets.get("transition_lora"),
                    detailer_lora_name=assets.get("detailer_lora"),
                    frame_guides=ltx_director_frame_guides,
                    video_combine_node=_available_comfy_node(object_info, "VHS_VideoCombine"),
                    available_nodes=set(object_info or {}),
                )
            elif request.preset.lower() == "wan":
                high_model_name = assets.get("wan_high_model")
                low_model_name = assets.get("wan_low_model")
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not high_model_name or not low_model_name:
                    missing_noise_models = []
                    if not high_model_name:
                        missing_noise_models.append("high-noise")
                    if not low_model_name:
                        missing_noise_models.append("low-noise")
                    selected_wan_model = assets.get("primary_model") or request.model_name or request.model_path or "Automatic"
                    raise ValueError(
                        "WAN 2.2 cannot resolve the "
                        + " and ".join(missing_noise_models)
                        + " base model. Put the missing WAN model in models/checkpoints/wan, models/unet or models/diffusion_models, "
                        "or select a compatible single-file WAN GGUF base model. Motion adapters/LoRAs/reference videos do not replace the WAN high/low base model. "
                        f"Selected model: {selected_wan_model}."
                    )
                if not text_encoder_name:
                    raise ValueError("WAN 2.2 requires a UMT5 text encoder in models/text_encoders.")
                if not vae_name:
                    raise ValueError("WAN 2.2 requires a Wan VAE in models/vae.")
                if not assets.get("clip_vision"):
                    raise ValueError("WAN 2.2 requires clip_vision_h.safetensors or a compatible CLIP Vision encoder in models/clip_vision.")
                wan_first_last_node = None
                wan_loop_cycle = _truthy((request.video or {}).get("wan_loop_cycle"))
                wan_motion_capture = _truthy((request.video or {}).get("wan_motion_capture_enabled"))
                wan_loop_source = str((request.video or {}).get("wan_loop_source") or "").strip().lower()
                if wan_motion_capture:
                    missing_motion_nodes = _wan_motion_missing_node_groups(object_info)
                    if missing_motion_nodes:
                        missing_labels = [
                            "/".join(str(name) for name in item.get("accepted_nodes", []) if name)
                            for item in missing_motion_nodes
                        ]
                        raise ValueError(
                            "WAN Motion Capture requires optional DWPose/Fun-Control nodes and models before generation: "
                            + ", ".join(missing_labels)
                            + ". Use the UI dependency prompt to install them; normal WAN remains available with Motion Capture off."
                        )
                    motion_model_signature = f"{high_model_name} {low_model_name}".lower()
                    motion_fun_control_ready = (
                        "fun" in motion_model_signature
                        and "control" in motion_model_signature
                    )
                    if not motion_fun_control_ready:
                        raise ValueError(
                            "WAN Motion Capture DWPose control requires Wan 2.2 Fun-Control compatible high/low models. "
                            f"The selected WAN models are not motion-control checkpoints: high={high_model_name}, low={low_model_name}. "
                            "Nexus will not auto-download base/checkpoint models; install/select compatible WAN Fun-Control or WAN Animate models, "
                            "or turn Motion Capture off to use normal WAN generation."
                        )
                if (
                    wan_loop_cycle
                    and reference_image_name
                    and not base_video_name
                    and (wan_loop_source == "start_frame_as_end_frame" or not reference_end_image_name)
                ):
                    reference_end_image_name = reference_image_name
                if base_video_name:
                    if not reference_image_name:
                        raise ValueError("WAN video2video requires an image reference.")
                    if not _available_comfy_node(object_info, "VHS_LoadVideo"):
                        raise ValueError("WAN video2video requires comfyui-videohelpersuite (VHS_LoadVideo).")
                    if not _available_comfy_node(object_info, "WanAnimateToVideo"):
                        raise ValueError("WAN video2video requires the native WanAnimateToVideo node.")
                    if wan_motion_capture:
                        prompt = build_basic_wan_motion_capture_workflow(
                            request,
                            high_model_name,
                            low_model_name,
                            text_encoder_name,
                            vae_name,
                            reference_image_name,
                            base_video_name,
                            clip_vision_name=assets.get("clip_vision"),
                        )
                    else:
                        prompt = build_basic_wan_video_reference_workflow(
                            request,
                            high_model_name,
                            low_model_name,
                            text_encoder_name,
                            vae_name,
                            reference_image_name,
                            base_video_name,
                            clip_vision_name=assets.get("clip_vision"),
                        )
                elif reference_end_image_name:
                    wan_first_last_node = _available_comfy_node(
                        object_info,
                        "WanFirstLastFrameToVideo",
                        "WanFirstLastFrameToVideoFunModel",
                    )
                    if wan_loop_cycle and not wan_first_last_node:
                        raise ValueError("WAN Loop Cycle requires WanFirstLastFrameToVideo support in the installed ComfyUI nodes.")
                if not base_video_name:
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
                    controlnet_name=assets.get("controlnet_model"),
                    controlnet_image_name=assets.get("controlnet_image"),
                    controlnet_category=assets.get("controlnet_category"),
                    available_nodes=set(object_info or {}),
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
                    controlnet_name=assets.get("controlnet_model"),
                    controlnet_image_name=assets.get("controlnet_image"),
                    controlnet_category=assets.get("controlnet_category"),
                )
            elif request.preset.lower() in {"ideogram4", "ideogram"}:
                checkpoint_name = assets.get("primary_model") or ""
                unconditional_name = assets.get("ideogram4_unconditional_model") or ""
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                if not checkpoint_name:
                    raise ValueError("Ideogram 4 requires ideogram4_fp8_scaled.safetensors in models/diffusion_models/ideogram4.")
                if not unconditional_name:
                    raise ValueError("Ideogram 4 requires ideogram4_unconditional_fp8_scaled.safetensors in models/diffusion_models/ideogram4.")
                if not text_encoder_name:
                    raise ValueError("Ideogram 4 requires qwen3vl_8b_fp8_scaled.safetensors in models/text_encoders.")
                if not vae_name:
                    raise ValueError("Ideogram 4 requires flux2-vae.safetensors in models/vae.")
                prompt = build_basic_ideogram4_workflow(
                    request,
                    checkpoint_name,
                    unconditional_name,
                    text_encoder_name,
                    vae_name,
                    reference_image_name=reference_image_name if request.activity == "img2img" else None,
                )
            elif request.preset.lower() == "flux":
                clip_l_name = assets.get("flux_clip_l")
                text_encoder_name = assets.get("text_encoder")
                vae_name = assets.get("vae")
                flux_family = assets.get("flux_family") or ""
                is_flux2 = str(flux_family).startswith("flux2")
                if is_flux2 and not assets.get("primary_model"):
                    raise ValueError("Flux.2 requires the selected Flux.2/Klein model file in models/checkpoints/flux, models/diffusion_models or models/unet.")
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
                    reference_image_names=reference_image_names,
                    mask_image_name=mask_image_name,
                    flux_family=flux_family,
                    controlnet_name=assets.get("controlnet_model"),
                    controlnet_image_name=assets.get("controlnet_image"),
                    available_nodes=set(object_info or {}),
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

        audio_normalization_bypassed = _bypass_missing_audio_normalization(prompt, object_info)
        if audio_normalization_bypassed:
            message = (
                "AudioVolumeNormalization is not loaded in ComfyUI; "
                "bypassing LTX audio volume normalization while keeping audio connected."
            )
            print(f"NEXUS BTA WARN {message} Nodes: {', '.join(audio_normalization_bypassed[:6])}", flush=True)
            if job_id:
                _update_generation_job(job_id, {"status": "building", "progress": 12, "message": message})
        ensure_inpaint_engine_route(prompt, request, available_nodes=set(object_info or {}))
        _materialize_ltx_director_audio(prompt)
        _apply_output_prefixes(prompt, request)
        _raise_if_generation_cancelled(job_id)

        def progress_callback(update: dict[str, Any]) -> None:
            if job_id:
                if _handle_cancelled_generation_progress(job_id, update):
                    return
                _update_generation_job(job_id, update)

        generation_started_at = datetime.now().timestamp()
        prompt_id, outputs = await comfy.run_workflow(
            prompt,
            progress_callback=progress_callback,
            timeout_seconds=_generation_timeout_seconds(request),
        )
        _raise_if_generation_cancelled(job_id)
        if not outputs:
            outputs = await _recover_outputs_from_history(prompt_id, generation_started_at)
        outputs = _cleanup_video_sidecar_images(outputs, generation_started_at)
        if not outputs:
            outputs = _cleanup_video_sidecar_images(_recent_output_files(generation_started_at - 300, limit=12), generation_started_at)
        if not outputs:
            await asyncio.sleep(1.0)
            outputs = _cleanup_video_sidecar_images(_recent_output_files(generation_started_at - 300, limit=20), generation_started_at)
        _apply_ltx_loop_cycle_seam(outputs, request)
        _normalize_ltx_start_end_motion(outputs, request, reference_image_names)
        _apply_ltx_reference_frame_lock(outputs, request, reference_image_names)
        _annotate_output_metadata(outputs, request, assets)
        _cleanup_generation_temp()
        quality_warnings = [
            str(item.get("warning"))
            for item in outputs
            if str(item.get("warning") or "").strip()
        ]
        response = GenerateResponse(
            job_id=prompt_id,
            prompt_id=prompt_id,
            status="completed",
            message=quality_warnings[0] if quality_warnings else "Generation completed.",
            outputs=outputs,
        )
        await _release_comfy_memory_if_idle()
        _schedule_comfy_idle_release()
        return response
    except Exception as exc:
        _cleanup_generation_temp()
        if request.preset.lower() == "model3d" and isinstance(exc, TimeoutError):
            try:
                await comfy.stop()
            except Exception:
                pass
        job_cancelled = bool(job_id and generation_jobs.get(job_id, {}).get("status") == "cancelled")
        if not job_cancelled:
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


def _ltx_preprocess_length(request: GenerateRequest) -> tuple[int, int]:
    video_options = request.video or {}
    fps = max(1, int(_number_or_none(video_options.get("fps")) or 8))
    requested_frames = _number_or_none(video_options.get("frames") or video_options.get("length"))
    if requested_frames is not None:
        length = max(2, int(round(requested_frames)))
    else:
        seconds = max(0.25, float(_number_or_none(video_options.get("seconds") or video_options.get("duration")) or 2.0))
        length = max(2, int(round(seconds * fps)))
    return fps, length


def _build_ltx_motion_preprocess_workflow(request: GenerateRequest, base_video_name: str, object_info: dict[str, Any]) -> dict[str, Any]:
    available = set(object_info or {})
    video_options = request.video or {}
    mode = str(video_options.get("motion_transfer_control_mode") or "pose").strip().lower()
    if mode not in {"pose", "canny", "depth", "camera"}:
        mode = "pose"
    fps, length = _ltx_preprocess_length(request)
    width = max(64, int(request.width or 512))
    height = max(64, int(request.height or 512))
    workflow: dict[str, Any] = {
        "1": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": base_video_name,
                "force_rate": float(fps),
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": length,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "LTXV",
            },
            "_meta": {"title": "Load Motion Reference Video"},
        }
    }
    image_ref: list[Any] = ["1", 0]
    if mode == "pose" and ("DWPreprocessor" in available or "OpenposePreprocessor" in available):
        pose_node = "DWPreprocessor" if "DWPreprocessor" in available else "OpenposePreprocessor"
        inputs: dict[str, Any] = {
            "image": ["1", 0],
            "detect_hand": "enable",
            "detect_body": "enable",
            "detect_face": "enable",
            "resolution": max(width, height),
            "scale_stick_for_xinsr_cn": "disable",
        }
        if pose_node == "DWPreprocessor":
            inputs["bbox_detector"] = "yolox_l.onnx"
            inputs["pose_estimator"] = "dw-ll_ucoco_384_bs5.torchscript.pt"
        workflow["2"] = {"class_type": pose_node, "inputs": inputs, "_meta": {"title": "Preprocess Motion Pose / DWPose"}}
        image_ref = ["2", 0]
    elif mode == "canny" and ("CannyEdgePreprocessor" in available or "Canny" in available):
        canny_node = "CannyEdgePreprocessor" if "CannyEdgePreprocessor" in available else "Canny"
        inputs = {"image": ["1", 0]}
        if canny_node == "CannyEdgePreprocessor":
            inputs.update({"low_threshold": 92, "high_threshold": 200, "resolution": max(width, height)})
        else:
            inputs.update({"low_threshold": 0.4, "high_threshold": 0.8})
        workflow["2"] = {"class_type": canny_node, "inputs": inputs, "_meta": {"title": "Preprocess Motion Canny"}}
        image_ref = ["2", 0]
    elif mode == "depth" and {"LoadVideoDepthAnythingModel", "VideoDepthAnythingProcess", "VideoDepthAnythingOutput"}.issubset(available):
        workflow["2"] = {"class_type": "LoadVideoDepthAnythingModel", "inputs": {"model": "video_depth_anything_vits.pth"}, "_meta": {"title": "Load Video Depth Anything"}}
        workflow["3"] = {"class_type": "VideoDepthAnythingProcess", "inputs": {"vda_model": ["2", 0], "images": ["1", 0], "input_size": 518, "max_res": max(width, height), "precision": "fp32"}, "_meta": {"title": "Preprocess Motion Depth"}}
        workflow["4"] = {"class_type": "VideoDepthAnythingOutput", "inputs": {"depths": ["3", 0], "colormap": "gray"}, "_meta": {"title": "Depth Preview Images"}}
        image_ref = ["4", 0]
    workflow["9"] = {
        "class_type": "VHS_VideoCombine",
        "inputs": {
            "images": image_ref,
            "frame_rate": float(fps),
            "loop_count": 0,
            "filename_prefix": f"Motion_Transfer/NEXUS_BTA_LTX23_MOTION_PREPROCESS_{mode}_{width}x{height}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": 16,
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        },
        "_meta": {"title": "Save LTX Motion Preprocess Preview"},
    }
    return workflow


async def _run_ltx_motion_preprocess_job(job_id: str, request: GenerateRequest) -> None:
    try:
        async with generation_lock:
            _update_generation_job(job_id, {"status": "preparing", "progress": 4, "message": "Preparing motion preprocess"}, force=True)
            if request.preset.lower() != "ltx" or request.activity != "img2img":
                raise ValueError("LTX Motion Preprocess requires LTX img2img.")
            base_video_name = _prepare_base_video(request)
            if not base_video_name:
                raise ValueError("Load a motion video before preprocessing.")
            await comfy.ensure_running()
            object_info = await comfy.object_info()
            workflow = _build_ltx_motion_preprocess_workflow(request, base_video_name, object_info)

            def progress_callback(update: dict[str, Any]) -> None:
                _update_generation_job(job_id, update)

            prompt_id, outputs = await comfy.run_workflow(workflow, progress_callback=progress_callback)
            if not outputs:
                outputs = await _recover_outputs_from_history(prompt_id, datetime.now().timestamp() - 300)
            outputs = _cleanup_video_sidecar_images(outputs, datetime.now().timestamp() - 300)
            preprocess_request = request.model_copy(deep=True)
            preprocess_request.video = dict(preprocess_request.video or {})
            preprocess_request.video["motion_preprocess"] = True
            _annotate_output_metadata(outputs, preprocess_request, {"primary_model": "LTX Motion Preprocess"})
            _update_generation_job(
                job_id,
                {
                    "prompt_id": prompt_id,
                    "status": "completed",
                    "progress": 100,
                    "message": "Motion preprocess completed.",
                    "outputs": outputs,
                    "completed_at": datetime.now().isoformat(timespec="seconds"),
                },
                force=True,
            )
    except Exception as exc:
        _update_generation_job(job_id, {"status": "failed", "progress": 100, "message": str(exc), "error": str(exc)}, force=True)


@app.post("/api/ltx/motion-preprocess/start")
async def ltx_motion_preprocess_start(request: GenerateRequest) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    generation_jobs[job_id] = {
        "job_id": job_id,
        "prompt_id": None,
        "status": "queued",
        "progress": 0,
        "message": "Queued motion preprocess.",
        "outputs": [],
        "error": None,
        "queue_position": len(_active_generation_jobs()) + 1,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "preset": request.preset,
        "workflow_id": "ltx_motion_preprocess",
        "_queued_monotonic": time.monotonic(),
    }
    _console_generation(generation_jobs[job_id], force=True)
    asyncio.create_task(_run_ltx_motion_preprocess_job(job_id, request))
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
        await comfy.interrupt(str(prompt_id) if prompt_id else None)
    except Exception:
        pass
    try:
        await comfy.clear_queue()
    except Exception:
        pass
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
    model_suffixes = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz"}
    for path in sorted(settings.output_dir.rglob("*"), key=lambda p: p.stat().st_mtime, reverse=True):
        if not path.is_file() or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".mp4", ".webm", ".mkv", ".mov", ".avi", *model_suffixes}:
            continue
        relative = path.relative_to(settings.output_dir).as_posix()
        url_path = quote(relative, safe="/")
        media_type = "3d" if path.suffix.lower() in model_suffixes else ("video" if path.suffix.lower() in {".mp4", ".webm", ".mkv", ".mov", ".avi"} else "image")
        if media_type == "image" and relative.replace("\\", "/").startswith("video/"):
            continue
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
