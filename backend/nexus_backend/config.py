from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
SETTINGS_PATH = CONFIG_DIR / "nexus_settings.json"


class RuntimeSettings(BaseModel):
    host: str = "127.0.0.1"
    nexus_port: int = 7861
    comfy_port: int = 8189
    auto_start_comfy: bool = True
    attention_backend: str = "auto"
    vram_policy: str = "balanced"
    precision: str = "auto"
    disable_xformers: bool = False
    enable_sage_attention: bool = False
    enable_flash_attention: bool = False
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
            data[key] = [Path(item) for item in data[key]]
    return data


def load_settings() -> NexusSettings:
    if SETTINGS_PATH.exists():
        raw = json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
        settings = NexusSettings(**_coerce_paths(raw))
    else:
        settings = NexusSettings()

    settings.ensure_directories()
    if not SETTINGS_PATH.exists():
        save_settings(settings)
    return settings


def save_settings(settings: NexusSettings) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    SETTINGS_PATH.write_text(
        settings.model_dump_json(indent=2),
        encoding="utf-8",
    )


def runtime_python(settings: NexusSettings) -> Path:
    if settings.comfy_python.exists():
        return settings.comfy_python
    return Path(sys.executable)
