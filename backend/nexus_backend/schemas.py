from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field


class ModelFile(BaseModel):
    name: str
    path: str
    relative_path: str
    category: str
    folder: str
    extension: str
    size_bytes: int
    modified: str
    tags: list[str] = Field(default_factory=list)
    source: Literal["managed", "reference"] = "managed"
    preview: str = ""


class ModelCatalog(BaseModel):
    root: str
    categories: dict[str, list[ModelFile]]
    total_files: int


class WorkflowSummary(BaseModel):
    id: str
    name: str
    path: str
    format: Literal["api", "ui", "unknown"]
    node_count: int
    class_types: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)


class WorkflowAnalysis(BaseModel):
    workflow: WorkflowSummary
    missing_nodes: list[str] = Field(default_factory=list)
    available_nodes: int = 0
    manager_suggestions: dict[str, list[str]] = Field(default_factory=dict)
    dependency_targets: list[str] = Field(default_factory=list)
    dependencies_installed: list[str] = Field(default_factory=list)
    dependency_errors: dict[str, str] = Field(default_factory=dict)
    visual_graph: dict[str, Any] = Field(default_factory=dict)
    workflow_settings: dict[str, Any] = Field(default_factory=dict)


class CustomNodeSummary(BaseModel):
    name: str
    path: str
    enabled: bool = True
    version: str = ""
    tags: list[str] = Field(default_factory=list)
    python_files: int = 0


class RuntimeHealth(BaseModel):
    nexus: str = "ok"
    comfy_running: bool
    comfy_url: str
    comfy_root_exists: bool
    comfy_python_exists: bool
    models_dir: str
    custom_nodes_dir: str


class DistilledLoraSelection(BaseModel):
    name: str = "None"
    strength: float = 1.0


class Img2ImgSettings(BaseModel):
    mode: str = "Image to Image"
    resize_mode: str = "Just Resize"
    aspect_ratio: str | None = None
    resize_percent: int = 100
    denoise: float = 0.75
    batch_count: int = 1
    mask_blur: int = 4
    mask_content: str = "Original"
    reference_image: str | None = None
    reference_images: list[str] = Field(default_factory=list)
    base_video: str | None = None
    mask_image: str | None = None


class ControlNetSettings(BaseModel):
    enabled: bool = False
    type: str = "canny"
    model: str = "Automatic"
    image: str | None = None
    strength: float = 0.75
    start_percent: float = 0.0
    end_percent: float = 1.0
    preprocessor: str = "Auto"
    low_threshold: float = 0.4
    high_threshold: float = 0.8
    balance: str = "Balanced"


class RuntimeOptions(BaseModel):
    vram_policy: str = "shared"
    gpu_memory_gb: float | None = None
    attention_backend: str = "auto"
    disable_xformers: bool = False
    precision: str = "auto"


class GenerateRequest(BaseModel):
    activity: str = "txt2img"
    workspace: str = "viewer"
    preset: str = "Anima"
    template: str | None = None
    workflow_id: str | None = None
    workflow_override: dict[str, Any] | None = None
    model_path: str | None = None
    model_name: str | None = None
    prompt: str = ""
    negative_prompt: str = ""
    width: int = 1024
    height: int = 576
    steps: int = 25
    cfg: float = 7.0
    sampler: str = "euler_ancestral"
    scheduler: str = "karras"
    seed: int = -1
    batch_size: int = 1
    denoise: float = 1.0
    vae: str = "Automatic"
    text_encoder: str = "Automatic"
    loras: list[dict[str, Any]] = Field(default_factory=list)
    distilled_loras: list[DistilledLoraSelection] = Field(default_factory=list)
    img2img: Img2ImgSettings = Field(default_factory=Img2ImgSettings)
    controlnet: ControlNetSettings = Field(default_factory=ControlNetSettings)
    video: dict[str, Any] = Field(default_factory=dict)
    director: dict[str, Any] = Field(default_factory=dict)
    runtime: RuntimeOptions = Field(default_factory=RuntimeOptions)


class GenerateResponse(BaseModel):
    job_id: str
    prompt_id: str | None = None
    status: str
    message: str
    outputs: list[dict[str, Any]] = Field(default_factory=list)


class ImportRequest(BaseModel):
    source: Path
    target_kind: Literal["models", "custom_nodes", "workflows", "comfy_core", "python_env"]
    mode: Literal["copy", "link"] = "copy"
    overwrite: bool = False


class WorkflowSaveRequest(BaseModel):
    name: str = "Nexus_Workflow"
    preset: str = "Anima"
    ui_state: dict[str, Any] = Field(default_factory=dict)
    workflow: dict[str, Any] | None = None


class DependencyInstallRequest(BaseModel):
    node_names: list[str] = Field(default_factory=list)
    all_enabled: bool = False


class CustomNodeUpdateRequest(BaseModel):
    version: str = ""


class PluginInstallRequest(BaseModel):
    url: str
    install_dependencies: bool = True


class CivitaiResolveRequest(BaseModel):
    url: str
    token: str | None = None
    target_kind: str = "auto"
    preset: str | None = None


class CivitaiDownloadRequest(BaseModel):
    url: str
    token: str | None = None
    target_kind: str = "auto"
    preset: str | None = None
    save_preview: bool = True


class CivitaiSearchRequest(BaseModel):
    query: str = ""
    token: str | None = None
    types: str = ""
    base_model: str = ""
    sort: str = "Newest"
    period: str = "AllTime"
    nsfw: bool = True
    limit: int = 24
    cursor: str | None = None
