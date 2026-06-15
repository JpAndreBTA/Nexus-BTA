from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "nexus_settings.json"
STARTUP_PATH = CONFIG_DIR / "nexus_startup.json"
STARTUP_ENV_PATH = CONFIG_DIR / "nexus_startup_env.cmd"


DEFAULT_MODEL_SOURCES: dict[str, list[Path]] = {
    "checkpoints": [],
    "diffusion_models": [],
    "unet": [],
    "vae": [],
    "text_encoders": [],
    "clip": [],
    "clip_vision": [],
    "loras": [],
    "controlnet": [],
    "model_patches": [],
    "upscale_models": [],
    "latent_upscale_models": [],
    "refine": [],
    "frame_interpolation": [],
    "video": [],
    "video_restore_models": [],
    "denoise_models": [],
    "face_restore_models": [],
    "background_removal": [],
    "embeddings": [],
    "animatediff_models": [],
    "animatediff_motion_lora": [],
    "style_models": [],
    "diffusers": [],
    "3d": [],
    "workflows": [],
}


class RuntimeSettings(BaseModel):
    host: str = "127.0.0.1"
    nexus_port: int = 7861
    comfy_port: int = 8189
    auto_start_comfy: bool = True
    attention_backend: str = "auto"
    vram_policy: str = "shared"
    gpu_memory_gb: float | None = None
    precision: str = "auto"
    disable_xformers: bool = False
    enable_sage_attention: bool = True
    enable_flash_attention: bool = False
    enable_pytorch_attention: bool = True
    acceleration_profile_version: int = 2
    restart_on_change: bool = False
    idle_unload_seconds: int = 90
    idle_stop_seconds: int = 300


class NexusSettings(BaseModel):
    project_root: Path = PROJECT_ROOT
    runtime_dir: Path = PROJECT_ROOT / "runtime"
    comfy_root: Path = PROJECT_ROOT / "runtime" / "ComfyUI"
    comfy_python: Path = PROJECT_ROOT / "runtime" / ".venv" / "Scripts" / "python.exe"
    models_dir: Path = PROJECT_ROOT / "models"
    custom_nodes_dir: Path = PROJECT_ROOT / "custom_nodes"
    workflows_dir: Path = PROJECT_ROOT / "workflows" / "comfyui"
    input_dir: Path = PROJECT_ROOT / "input"
    output_dir: Path = PROJECT_ROOT / "output"
    temp_dir: Path = PROJECT_ROOT / "temp"
    user_dir: Path = PROJECT_ROOT / "user"
    reference_model_sources: list[Path] = Field(
        default_factory=lambda: [Path(r"C:\ComfyUpdate\models")]
    )
    model_sources: dict[str, list[Path]] = Field(
        default_factory=lambda: {key: list(paths) for key, paths in DEFAULT_MODEL_SOURCES.items()}
    )
    reference_custom_node_sources: list[Path] = Field(
        default_factory=lambda: [Path(r"C:\ComfyUpdate\custom_nodes")]
    )
    reference_workflow_sources: list[Path] = Field(
        default_factory=lambda: [Path(r"C:\Users\jpzin\OneDrive\Documentos\Comfy work")]
    )
    comfy_core_sources: list[Path] = Field(
        default_factory=lambda: [Path(r"C:\ComfyUI\resources\ComfyUI")]
    )
    python_env_sources: list[Path] = Field(
        default_factory=lambda: [Path(r"C:\ComfyUpdate\.venv")]
    )
    supported_model_extensions: list[str] = Field(
        default_factory=lambda: [
            ".safetensors",
            ".ckpt",
            ".pt",
            ".pth",
            ".bin",
            ".gguf",
            ".onnx",
        ]
    )
    runtime: RuntimeSettings = Field(default_factory=RuntimeSettings)

    def ensure_directories(self) -> None:
        for path in [
            CONFIG_DIR,
            self.runtime_dir,
            self.models_dir,
            self.custom_nodes_dir,
            self.workflows_dir,
            self.input_dir,
            self.output_dir,
            self.temp_dir,
            self.user_dir,
        ]:
            path.mkdir(parents=True, exist_ok=True)


def _coerce_paths(data: dict[str, Any]) -> dict[str, Any]:
    path_keys = {
        "project_root",
        "runtime_dir",
        "comfy_root",
        "comfy_python",
        "models_dir",
        "custom_nodes_dir",
        "workflows_dir",
        "input_dir",
        "output_dir",
        "temp_dir",
        "user_dir",
    }
    list_path_keys = {
        "reference_model_sources",
        "reference_custom_node_sources",
        "reference_workflow_sources",
        "comfy_core_sources",
        "python_env_sources",
    }
    for key in path_keys:
        if key in data and data[key]:
            data[key] = Path(data[key])
    for key in list_path_keys:
        if key in data and data[key]:
            data[key] = coerce_path_list(data[key])
    if isinstance(data.get("model_sources"), dict):
        data["model_sources"] = {
            str(key): [Path(item) for item in value]
            for key, value in data["model_sources"].items()
            if isinstance(value, list)
        }
    return data


def coerce_path_list(value: Any) -> list[Path]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        text = str(value).strip()
        return [Path(text)] if text else []
    if isinstance(value, list):
        if value and all(isinstance(item, str) and len(item) == 1 for item in value):
            joined = "".join(value).strip()
            return [Path(joined)] if joined else []
        return [Path(item) for item in value if str(item).strip()]
    return []


def load_settings() -> NexusSettings:
    migrated = False
    if SETTINGS_PATH.exists():
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8-sig"))
        settings = NexusSettings(**_coerce_paths(raw))
    else:
        settings = NexusSettings()

    runtime_policy = str(settings.runtime.vram_policy or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    if runtime_policy in {"gpu", "gpuonly", "onlygpu", "cudaonly"}:
        if settings.runtime.vram_policy != "gpu_only":
            settings.runtime.vram_policy = "gpu_only"
            migrated = True
    elif settings.runtime.vram_policy != "shared":
        settings.runtime.vram_policy = "shared"
    if int(getattr(settings.runtime, "acceleration_profile_version", 0) or 0) < 2:
        attention = str(settings.runtime.attention_backend or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
        old_forced_sdpa = attention in {"pytorch", "pytorchsdpa", "sdpa"} and bool(settings.runtime.disable_xformers)
        old_empty_auto = attention in {"", "auto"} and bool(settings.runtime.disable_xformers) and not bool(settings.runtime.enable_sage_attention)
        if old_forced_sdpa or old_empty_auto:
            settings.runtime.attention_backend = "sage"
            settings.runtime.disable_xformers = False
            settings.runtime.enable_sage_attention = True
            settings.runtime.enable_flash_attention = False
        settings.runtime.acceleration_profile_version = 2
        migrated = True

    for key, value in DEFAULT_MODEL_SOURCES.items():
        if key not in settings.model_sources:
            settings.model_sources[key] = list(value)
            migrated = True

    _apply_environment_overrides(settings)
    settings.ensure_directories()
    if migrated or not SETTINGS_PATH.exists():
        save_settings(settings)
    return settings


def _apply_environment_overrides(settings: NexusSettings) -> None:
    path_env = {
        "NEXUS_MODELS_DIR": "models_dir",
        "NEXUS_COMFY_ROOT": "comfy_root",
        "NEXUS_COMFY_PYTHON": "comfy_python",
        "NEXUS_CUSTOM_NODES_DIR": "custom_nodes_dir",
        "NEXUS_WORKFLOWS_DIR": "workflows_dir",
        "NEXUS_INPUT_DIR": "input_dir",
        "NEXUS_OUTPUT_DIR": "output_dir",
        "NEXUS_TEMP_DIR": "temp_dir",
        "NEXUS_USER_DIR": "user_dir",
    }
    for env_name, attr_name in path_env.items():
        value = os.environ.get(env_name, "").strip()
        if value:
            setattr(settings, attr_name, Path(value))

    port_value = os.environ.get("NEXUS_COMFY_PORT", "").strip()
    if port_value.isdigit():
        settings.runtime.comfy_port = int(port_value)


def save_settings(settings: NexusSettings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )


def _same_path(left: Path | str | None, right: Path | str | None) -> bool:
    if left is None or right is None:
        return False
    try:
        return Path(left).resolve() == Path(right).resolve()
    except OSError:
        return str(left).strip().rstrip("\\/").lower() == str(right).strip().rstrip("\\/").lower()


def sync_startup_model_path(settings: NexusSettings, previous_models_dir: Path | None = None) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {}
    if STARTUP_PATH.exists():
        try:
            loaded = json.loads(STARTUP_PATH.read_text(encoding="utf-8-sig"))
            if isinstance(loaded, dict):
                state = loaded
        except (json.JSONDecodeError, OSError):
            state = {}

    selected_models_dir = str(settings.models_dir)
    model_sources = {
        str(key): [str(path) for path in paths]
        for key, paths in settings.model_sources.items()
    }
    previous_sources = state.get("model_sources") if isinstance(state.get("model_sources"), dict) else {}
    relevant_source_keys = ("checkpoints", "diffusion_models", "unet")
    relevant_sources_changed = any(
        [str(item) for item in previous_sources.get(key, [])] != model_sources.get(key, [])
        for key in relevant_source_keys
    )
    existing_selection_matches = _same_path(state.get("selected_models_dir"), settings.models_dir)
    model_path_changed = previous_models_dir is not None and not _same_path(previous_models_dir, settings.models_dir)

    state["model_path_prompted"] = True
    state["selected_models_dir"] = selected_models_dir
    state["models_dir"] = selected_models_dir
    state["model_sources"] = model_sources
    state["reference_model_sources"] = [str(path) for path in settings.reference_model_sources]
    state["custom_nodes_dir"] = str(settings.custom_nodes_dir)
    state["reference_custom_node_sources"] = [str(path) for path in settings.reference_custom_node_sources]
    state["comfy_root"] = str(settings.comfy_root)
    state["comfy_python"] = str(settings.comfy_python)
    state["workflows_dir"] = str(settings.workflows_dir)
    state["reference_workflow_sources"] = [str(path) for path in settings.reference_workflow_sources]
    state["model_path_source"] = "ui_settings"
    state["saved_at"] = datetime.now().isoformat(timespec="seconds")
    state["last_checked_at"] = state["saved_at"]

    if model_path_changed or not existing_selection_matches or relevant_sources_changed:
        state["download_optional_models"] = False
        state["download_choice_saved_at"] = state["saved_at"]
    elif "download_optional_models" not in state:
        state["download_optional_models"] = False

    STARTUP_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")
    allow_downloads = "1" if bool(state.get("download_optional_models")) else "0"
    STARTUP_ENV_PATH.write_text(
        "\n".join(
            [
                "@echo off",
                f'set "NEXUS_MODELS_DIR={selected_models_dir}"',
                f'set "NEXUS_ALLOW_MODEL_DOWNLOADS={allow_downloads}"',
                f'set "NEXUS_COMFY_ROOT={settings.comfy_root}"',
                f'set "NEXUS_COMFY_PYTHON={settings.comfy_python}"',
                f'set "NEXUS_CUSTOM_NODES_DIR={settings.custom_nodes_dir}"',
                "",
            ]
        ),
        encoding="ascii",
    )


def runtime_python(settings: NexusSettings) -> Path:
    if settings.comfy_python.exists():
        return settings.comfy_python
    return Path(sys.executable)
