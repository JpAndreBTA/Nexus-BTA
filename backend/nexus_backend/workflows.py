from __future__ import annotations

import json
import importlib.util
import math
import random
import re
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import NexusSettings
from .dependencies import custom_nodes_for_workflow, install_custom_node_dependencies, manager_suggestions_for_nodes
from .ideogram4_prompt import is_ideogram4_prompt_json, parse_ideogram4_prompt_json, ideogram4_prompt_json_text
from .schemas import GenerateRequest, WorkflowAnalysis, WorkflowSaveRequest, WorkflowSummary


SAMPLER_ALIASES = {
    "Euler Ancestral": "euler_ancestral",
    "Euler": "euler",
    "Heun": "heun",
    "Heun++": "heunpp2",
    "LMS": "lms",
    "DPM2": "dpm_2",
    "DPM2 Ancestral": "dpm_2_ancestral",
    "DPM Fast": "dpm_fast",
    "DPM Adaptive": "dpm_adaptive",
    "DPM++ 2S Ancestral": "dpmpp_2s_ancestral",
    "DPM++ 2M SDE": "dpmpp_2m_sde",
    "DPM++ 2M SDE Heun": "dpmpp_2m_sde_heun",
    "DPM++ 2M": "dpmpp_2m",
    "DPM++ SDE": "dpmpp_sde",
    "DPM++ 3M SDE": "dpmpp_3m_sde",
    "DPM++ 3M SDE GPU": "dpmpp_3m_sde_gpu",
    "DDIM": "ddim",
    "UniPC": "uni_pc",
    "UniPC BH2": "uni_pc_bh2",
    "LCM": "lcm",
    "Euler CFG++": "euler_cfg_pp",
    "Euler Ancestral CFG++": "euler_ancestral_cfg_pp",
    "ER SDE": "er_sde",
    "TCD": "tcd",
    "DEIS": "deis",
    "IPNDM": "ipndm",
    "IPNDM V": "ipndm_v",
    "Restart": "restart",
    "FlowMatch Euler": "euler",
    "Res Multistep": "res_multistep",
}

SCHEDULER_ALIASES = {
    "Karras": "karras",
    "Normal": "normal",
    "Exponential": "exponential",
    "SGM Uniform": "sgm_uniform",
    "Simple": "simple",
    "DDIM Uniform": "ddim_uniform",
    "Beta": "beta",
    "Quadratic": "quadratic",
    "Linear Quadratic": "linear_quadratic",
    "AYS SD1": "ays_sd1",
    "AYS SDXL": "ays_sdxl",
    "AYS SVD": "ays_svd",
    "GITS": "gits",
    "Align Your Steps": "align_your_steps",
}

UI_HELPER_NODE_TYPES = {
    "Note",
    "MarkdownNote",
    "PrimitiveNode",
    "SetNode",
    "GetNode",
    "Anything Everywhere",
    "Combo Clone",
    "Fast Groups Bypasser (rgthree)",
    "Fast Groups Muter (rgthree)",
    "Fast Actions Button (rgthree)",
    "Label (rgthree)",
    "Bookmark (rgthree)",
    "FancyTimerNode",
}

LTX_OMNICINE_LORA_NAME = "ltx\\Singularity LTX-2.3  OmniCine Preview v0.1.safetensors"
LTX_OMNICINE_DEFAULT_STRENGTH = 0.75
LTX_TRANSITION_LORA_NAME = "ltx_transition\\ltx2.3-transition.safetensors"
LTX_TRANSITION_DEFAULT_STRENGTH = 1.0
LTX_TRANSITION_TRIGGER = "zhuanchang"
LTX_TRANSITION_PROMPT_HINT = "seamless continuous transition motion that completes at the final frame"
LTX_DISTILLED_CONDSAFE_DEFAULT_STRENGTH = 0.80
LTX_DISTILLED_384_DEFAULT_STRENGTH = 0.50
LTX_DISTILLED_8_STEP_SIGMAS = "1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0"
LTX_TRANSITION_8_STEP_SIGMAS = "1.0, 0.9375, 0.8125, 0.5625, 0.25, 0.0"
LTX_UPSCALE_REFINER_SIGMAS = "0.85, 0.7250, 0.4219, 0.0"
LTX_TRANSITION_REFINER_SIGMAS = "0.85, 0.45, 0.0"


def normalize_sampler(value: str) -> str:
    return SAMPLER_ALIASES.get(value, value).lower().replace(" ", "_")


def normalize_scheduler(value: str) -> str:
    return SCHEDULER_ALIASES.get(value, value).lower().replace(" ", "_")


def _clip_loader_node(clip_name: str, clip_type: str, title: str) -> dict[str, Any]:
    if str(clip_name or "").lower().endswith(".gguf"):
        return {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": clip_name, "type": clip_type},
            "_meta": {"title": title},
        }
    return {
        "class_type": "CLIPLoader",
        "inputs": {"clip_name": clip_name, "type": clip_type, "device": "default"},
        "_meta": {"title": title},
    }


def is_anima_qwen35_text_encoder(text_encoder_name: str | None) -> bool:
    name = Path(str(text_encoder_name or "")).name.lower()
    return "anima2bqwen" in name or "qwen35" in name or "qwen3.5" in name


def _anima_clip_loader_node(text_encoder_name: str, title: str) -> dict[str, Any]:
    if is_anima_qwen35_text_encoder(text_encoder_name):
        return {
            "class_type": "LoadQwen35AnimaCLIP",
            "inputs": {
                "clip_name": text_encoder_name,
                "use_calibration": True,
                "use_alignment": True,
                "alignment_strength": 0.0,
                "output_scale": 1.0,
            },
            "_meta": {"title": "Anima Qwen3.5 Text Encoder"},
        }
    return _clip_loader_node(text_encoder_name, "stable_diffusion", title)


def workflow_id_from_path(path: Path) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", path.stem).strip("-").lower()
    return slug or path.stem.lower()


class WorkflowRegistry:
    def __init__(self, settings: NexusSettings):
        self.settings = settings

    def ensure_model3d_workflow_aliases(self) -> int:
        self.settings.workflows_dir.mkdir(parents=True, exist_ok=True)
        aliases = [
            (
                self.settings.workflows_dir / "model3d_trellis2_meshwithtexturing_multiview.json",
                self.settings.workflows_dir / "model3d_trellis2_meshwithvoxel_texturing_multiview.json",
            ),
        ]
        count = 0
        for source, target in aliases:
            if source.exists() and not target.exists():
                shutil.copy2(source, target)
                count += 1
        return count

    def import_reference_workflows(self) -> int:
        count = 0
        self.settings.workflows_dir.mkdir(parents=True, exist_ok=True)
        for source in self.settings.reference_workflow_sources:
            if not source.exists():
                continue
            for item in source.glob("*.json"):
                target = self.settings.workflows_dir / item.name
                if not target.exists():
                    shutil.copy2(item, target)
                    count += 1
        count += self.ensure_model3d_workflow_aliases()
        return count

    def list_workflows(self) -> list[WorkflowSummary]:
        workflows: list[WorkflowSummary] = []
        roots = [
            self.settings.project_root / "workflows" / "nexus_base",
            self.settings.workflows_dir,
            *self.settings.reference_workflow_sources,
        ]
        seen: set[Path] = set()
        for root in roots:
            if not root.exists():
                continue
            for path in sorted(root.glob("*.json"), key=lambda p: p.name.lower()):
                resolved = path.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                workflows.append(self.summarize(path))
        return workflows

    def import_workflow_file(self, filename: str, content: bytes) -> WorkflowSummary:
        self.settings.workflows_dir.mkdir(parents=True, exist_ok=True)
        return self._write_workflow_file(self.settings.workflows_dir, filename, content)

    def load_workflow_file(self, filename: str, content: bytes) -> WorkflowSummary:
        target_dir = self.settings.temp_dir / "loaded_workflows"
        target_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", Path(filename).name).strip() or "workflow.json"
        target = target_dir / safe_name
        target.write_bytes(content)
        return self.summarize(target)

    def _write_workflow_file(self, root: Path, filename: str, content: bytes) -> WorkflowSummary:
        root.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", Path(filename).name).strip() or "workflow.json"
        if not safe_name.lower().endswith(".json"):
            safe_name += ".json"
        target = root / safe_name
        stem = target.stem
        counter = 1
        while target.exists():
            target = root / f"{stem}_{counter}.json"
            counter += 1
        target.write_bytes(content)
        return self.summarize(target)

    def save_workflow(self, request: WorkflowSaveRequest) -> WorkflowSummary:
        self.settings.workflows_dir.mkdir(parents=True, exist_ok=True)
        safe_name = re.sub(r"[^a-zA-Z0-9._ -]+", "_", request.name).strip("_") or "Nexus_Workflow"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target = self.settings.workflows_dir / f"{safe_name}_{timestamp}.json"
        payload = request.workflow or {
            "nexus_bta": {
                "preset": request.preset,
                "saved_at": timestamp,
                "ui_state": request.ui_state,
            }
        }
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return self.summarize(target)

    def analyze_workflow(
        self,
        workflow: WorkflowSummary,
        object_info: dict[str, Any] | None = None,
        install_dependencies: bool = False,
    ) -> WorkflowAnalysis:
        object_info = object_info or {}
        available = set(object_info.keys())
        missing = sorted({
            class_type
            for class_type in workflow.class_types
            if class_type not in available and not _is_ui_helper_node(class_type)
        })
        suggestions = manager_suggestions_for_nodes(self.settings, missing)
        dependency_targets = custom_nodes_for_workflow(self.settings, workflow.class_types, suggestions)
        installed: list[str] = []
        errors: dict[str, str] = {}
        if install_dependencies and dependency_targets:
            installed, errors = install_custom_node_dependencies(self.settings, node_names=dependency_targets)
        return WorkflowAnalysis(
            workflow=workflow,
            missing_nodes=missing,
            available_nodes=len(available),
            manager_suggestions=suggestions,
            dependency_targets=dependency_targets,
            dependencies_installed=installed,
            dependency_errors=errors,
            visual_graph=workflow_visual_graph(Path(workflow.path)),
            workflow_settings=workflow_settings(Path(workflow.path), object_info=object_info),
        )

    def find(self, workflow_id: str | None, preset: str | None = None) -> Path | None:
        workflows = self.list_workflows()
        if workflow_id:
            for workflow in workflows:
                if workflow.id == workflow_id:
                    return Path(workflow.path)
            temp_dir = self.settings.temp_dir / "loaded_workflows"
            if temp_dir.exists():
                for path in temp_dir.glob("*.json"):
                    if workflow_id_from_path(path) == workflow_id:
                        return path
        if preset:
            preset_lower = preset.lower()
            preferred = {
                "ltx": [],
                "anima": [],
                "wan": ["wan"],
                "flux": [],
                "qwen": [],
                "model3d": ["meshwithvoxel", "meshwithtexturing", "trellis", "3d", "meshonly"],
                "zimageturbo": ["z-image-turbo", "zimage-turbo", "zimage"],
                "zimage": ["z-image-turbo", "zimage-turbo", "zimage"],
                "lumina": ["lumina"],
            }.get(preset_lower, [])
            for token in preferred:
                for workflow in workflows:
                    if token in workflow.id:
                        return Path(workflow.path)
        return None

    def summarize(self, path: Path) -> WorkflowSummary:
        data = json.loads(path.read_text(encoding="utf-8"))
        fmt = detect_workflow_format(data)
        classes = class_types(data, fmt)
        tags = sorted(
            {
                tag
                for tag in ["ideogram4", "ideogram", "ltx", "anima", "wan", "flux", "qwen", "zimage", "z-image", "trellis", "3d", "mesh", "gguf", "i2v", "t2v"]
                if tag in path.name.lower() or any(tag in cls.lower() for cls in classes)
            }
        )
        count = len(data.get("nodes", [])) if fmt == "ui" else len(data) if isinstance(data, dict) else 0
        return WorkflowSummary(
            id=workflow_id_from_path(path),
            name=path.stem,
            path=str(path),
            format=fmt,
            node_count=count,
            class_types=classes,
            tags=tags,
        )

    def load_api_workflow(
        self,
        path: Path,
        request: GenerateRequest,
        object_info: dict[str, Any] | None = None,
        assets: dict[str, str] | None = None,
    ) -> dict[str, Any]:
        data = json.loads(path.read_text(encoding="utf-8"))
        fmt = detect_workflow_format(data)
        if fmt == "api":
            api = deepcopy(data)
            api = {
                str(node_id): node
                for node_id, node in api.items()
                if isinstance(node, dict) and "class_type" in node
            }
        elif fmt == "ui":
            api = convert_ui_to_api(data, object_info or {})
        else:
            raise ValueError(f"Unsupported workflow format: {path}")
        return patch_workflow(api, request, assets=assets)


def detect_workflow_format(data: Any) -> str:
    if isinstance(data, dict) and isinstance(data.get("nodes"), list):
        return "ui"
    if isinstance(data, dict) and any(isinstance(v, dict) and "class_type" in v for v in data.values()):
        return "api"
    return "unknown"


def class_types(data: Any, fmt: str) -> list[str]:
    if fmt == "ui":
        ui_nodes = list(data.get("nodes", []) or [])
        for subgraph in data.get("definitions", {}).get("subgraphs") or []:
            if isinstance(subgraph, dict):
                ui_nodes.extend(subgraph.get("nodes") or [])
        values = [node.get("type") or node.get("class_type") for node in ui_nodes if isinstance(node, dict)]
    elif fmt == "api":
        values = [node.get("class_type") for node in data.values() if isinstance(node, dict)]
    else:
        values = []
    unique = []
    for value in values:
        if value and value not in unique:
            unique.append(value)
    return unique


def workflow_visual_graph(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"nodes": [], "links": [], "width": 1300, "height": 720}
    fmt = detect_workflow_format(data)
    if fmt == "ui":
        return _ui_workflow_graph(data)
    if fmt == "api":
        return _api_workflow_graph(data)
    return {"nodes": [], "links": [], "width": 1300, "height": 720}


def workflow_settings(path: Path, object_info: dict[str, Any] | None = None) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    fmt = detect_workflow_format(data)
    sources: list[Any] = [] if fmt == "ui" else [data]

    result: dict[str, Any] = {}
    text_hints: list[str] = []

    def capture(key: str, value: Any, *, force: bool = False) -> None:
        lower = key.lower()
        if isinstance(value, (dict, list)):
            return
        text_value = str(value)
        haystack = f"{lower} {text_value.lower()}"
        text_hints.append(haystack)

        def set_number(name: str) -> None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                return
            if name in result and not force:
                if name == "fps" and 0 < number <= 240 and number > float(result.get(name) or 0):
                    result[name] = int(number) if number.is_integer() else number
                return
            result[name] = int(number) if number.is_integer() else number

        if lower in {"width", "empty_latent_width", "image_width", "custom_width"}:
            set_number("width")
        elif lower in {"height", "empty_latent_height", "image_height", "custom_height"}:
            set_number("height")
        elif lower in {"steps", "num_steps"}:
            set_number("steps")
        elif lower in {"cfg", "cfg_scale", "guidance_scale"}:
            set_number("cfg")
        elif lower in {"seed", "noise_seed"}:
            set_number("seed")
        elif lower in {"fps", "frame_rate", "framerate"}:
            set_number("fps")
        elif lower in {"frames", "num_frames", "frame_count", "frame_number", "frames_number", "num_video_frames", "video_frames", "length", "duration_frames"}:
            set_number("frames")
        elif lower in {"duration", "seconds", "duration_seconds", "video_seconds"}:
            set_number("seconds")
        elif lower in {"active_audio", "audio_enabled", "enable_audio"}:
            result["active_audio"] = str(value).lower() not in {"false", "0", "off", "none", "no"}
        elif lower in {"omnicine", "omnicine_enabled"}:
            result["omnicine_enabled"] = str(value).lower() not in {"false", "0", "off", "none", "no"}
        elif lower in {"sampler_name", "sampler"} and "sampler" not in result:
            result["sampler"] = text_value
        elif lower in {"scheduler", "scheduler_name", "schedule", "schedule_type"} and "scheduler" not in result:
            result["scheduler"] = text_value
        elif "vae" in lower and _looks_like_model_file(text_value) and "vae" not in result:
            result["vae"] = Path(text_value).name
        elif ("clip" in lower or "encoder" in lower or "text_encoder" in lower) and _looks_like_model_file(text_value) and "text_encoder" not in result:
            result["text_encoder"] = Path(text_value).name
        elif "lora" in lower and _looks_like_model_file(text_value):
            loras = result.setdefault("loras", [])
            if text_value not in loras:
                loras.append(Path(text_value).name)

    def walk(value: Any) -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                capture(str(key), item)
                walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    def capture_ui_widgets(value: Any) -> None:
        if not isinstance(value, dict):
            return
        ui_nodes = list(value.get("nodes") or [])
        for subgraph in value.get("definitions", {}).get("subgraphs") or []:
            if isinstance(subgraph, dict):
                ui_nodes.extend(subgraph.get("nodes") or [])
        for node in ui_nodes:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type", ""))
            title = str(node.get("title") or node.get("type") or "").lower()
            widgets = node.get("widgets_values")
            if isinstance(widgets, dict):
                for key, item in widgets.items():
                    capture(str(key), item)
                continue
            if isinstance(widgets, list):
                order = _known_ui_widget_order(node_type)
                if order:
                    for index, item in enumerate(widgets):
                        key = order[index] if index < len(order) else ""
                        if key:
                            force_dimension = "latentvideo" in node_type.lower() and key in {"width", "height", "length"}
                            capture(key, item, force=force_dimension)
                            if key == "value":
                                if "fps" in title or "frame rate" in title:
                                    capture("fps", item, force=True)
                                elif "second" in title or "duration" in title:
                                    capture("seconds", item, force=True)
                                elif "frame" in title or "length" in title:
                                    capture("frames", item, force=True)
                for item in widgets:
                    if isinstance(item, str) and _looks_like_model_file(item):
                        clean_name = Path(item.replace("\\", "/")).name
                        haystack = f"{node_type} {title} {item}".lower()
                        if "audio" in haystack and "vae" in haystack:
                            result["audio_vae"] = clean_name
                        elif "video" in haystack and "vae" in haystack:
                            result["video_vae"] = clean_name
                        elif "upscale" in haystack and ("latent" in haystack or "spatial" in haystack):
                            result["latent_upscale"] = clean_name
                        elif "vae" in haystack and "vae" not in result:
                            result["vae"] = clean_name
                        elif ("clip" in haystack or "encoder" in haystack) and "text_encoder" not in result:
                            result["text_encoder"] = clean_name
                walk(widgets)
                continue
            if widgets is not None:
                widget_text = str(widgets)
                lower_widget = widget_text.lower()
                haystack = f"{node_type} {title} {lower_widget}".lower()
                if _looks_like_model_file(widget_text):
                    clean_name = Path(widget_text.replace("\\", "/")).name
                    if "audio" in haystack and "vae" in haystack:
                        result["audio_vae"] = clean_name
                    elif "video" in haystack and "vae" in haystack:
                        result["video_vae"] = clean_name
                    elif "upscale" in haystack and ("latent" in haystack or "spatial" in haystack):
                        result["latent_upscale"] = clean_name
                    elif "vae" in haystack and "vae" not in result:
                        result["vae"] = clean_name
                    elif ("clip" in haystack or "encoder" in haystack) and "text_encoder" not in result:
                        result["text_encoder"] = clean_name
                if "ksamplerselect" in haystack and widget_text:
                    capture("sampler_name", widgets, force=True)
                elif "cfgguider" in haystack:
                    capture("cfg", widgets, force=True)
                if "fps" in title or "frame rate" in title:
                    capture("fps", widgets, force=True)
                elif "second" in title or "duration" in title:
                    capture("seconds", widgets, force=True)
                elif "frame" in title or "length" in title:
                    capture("frames", widgets, force=True)

    def prompt_sources(value: Any) -> list[dict[str, Any]]:
        prompts: list[dict[str, Any]] = []
        if isinstance(value, dict):
            if all(isinstance(item, dict) and "class_type" in item for item in value.values()):
                prompts.append(value)
            extra_prompt = value.get("extra", {}).get("prompt")
            if isinstance(extra_prompt, dict):
                prompts.append(extra_prompt)
        return prompts

    def capture_prompt_text(prompt: dict[str, Any]) -> None:
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            class_type = str(node.get("class_type") or node.get("type") or "")
            meta = node.get("_meta") if isinstance(node.get("_meta"), dict) else {}
            title = str(meta.get("title") or node.get("title") or class_type)
            haystack = f"{title} {class_type}".lower()
            if not any(token in haystack for token in ("cliptextencode", "textencode", "prompt")):
                continue
            inputs = node.get("inputs") or {}
            if not isinstance(inputs, dict):
                continue
            text = None
            for key in ("text", "prompt", "positive", "negative"):
                value = inputs.get(key)
                if isinstance(value, str) and value.strip():
                    text = value
                    break
            if not text:
                continue
            if "negative" in haystack:
                result["negative_prompt"] = text
            elif "prompt" not in result:
                result["prompt"] = text

    def capture_ui_prompt_text(value: Any) -> None:
        if not isinstance(value, dict):
            return
        ui_nodes = list(value.get("nodes") or [])
        for subgraph in value.get("definitions", {}).get("subgraphs") or []:
            if isinstance(subgraph, dict):
                ui_nodes.extend(subgraph.get("nodes") or [])
        for node in ui_nodes:
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("type", ""))
            title = str(node.get("title") or node_type)
            haystack = f"{title} {node_type}".lower()
            if not any(token in haystack for token in ("cliptextencode", "textencode", "prompt")):
                continue
            widgets = node.get("widgets_values")
            text = None
            if isinstance(widgets, dict):
                for key in ("text", "prompt", "positive", "negative"):
                    if isinstance(widgets.get(key), str) and widgets[key].strip():
                        text = widgets[key]
                        break
            elif isinstance(widgets, list):
                text = next((item for item in widgets if isinstance(item, str) and item.strip() and not _looks_like_model_file(item)), None)
            if not text:
                continue
            if "negative" in haystack:
                result["negative_prompt"] = text
            elif "prompt" not in result:
                result["prompt"] = text

    def capture_linked_constants(prompt: dict[str, Any]) -> None:
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs") or {}
            if not isinstance(inputs, dict):
                continue
            for key, item in inputs.items():
                if not (isinstance(item, list) and item):
                    continue
                source = prompt.get(str(item[0]))
                if not isinstance(source, dict):
                    continue
                source_inputs = source.get("inputs") or {}
                if not isinstance(source_inputs, dict):
                    continue
                if key in {"sampler", "sampler_name"} and "sampler_name" in source_inputs:
                    capture("sampler_name", source_inputs["sampler_name"], force=True)
                    continue
                if str(key).lower() not in {
                    "width",
                    "height",
                    "empty_latent_width",
                    "empty_latent_height",
                    "steps",
                    "cfg",
                    "cfg_scale",
                    "seed",
                    "noise_seed",
                    "fps",
                    "frame_rate",
                    "framerate",
                    "frames",
                    "num_frames",
                    "frame_count",
                    "frames_number",
                    "length",
                    "seconds",
                    "duration",
                    "duration_seconds",
                    "video_seconds",
                }:
                    continue
                for constant_key in ("value", "number", "int", "float"):
                    if constant_key in source_inputs:
                        capture(str(key), source_inputs[constant_key], force=str(key).lower() in {"fps", "frame_rate", "framerate", "frames", "num_frames", "frame_count", "frames_number", "length"})
                        break

    if fmt == "ui":
        capture_ui_widgets(data)
        capture_ui_prompt_text(data)
        for prompt in prompt_sources(data):
            walk(prompt)
            capture_prompt_text(prompt)
            capture_linked_constants(prompt)
    for source in sources:
        walk(source)
        capture_ui_widgets(source)
        for prompt in prompt_sources(source):
            capture_prompt_text(prompt)
            capture_linked_constants(prompt)
    combined = " ".join(text_hints + class_types(data, fmt)).lower()
    result["is_video"] = any(token in combined for token in ["video", "frames", "fps", "ltx", "wan", "animatediff", "vhs_", "frame_rate"])
    if "ideogram4" in combined or "ideogram 4" in combined or "ideogram-4" in combined or "ideogram4promptbuilderkj" in combined:
        result["preset"] = "Ideogram4"
        result["is_video"] = False
        result.setdefault("scheduler", "simple")
        result.setdefault("sampler", "euler")
        result.setdefault("steps", 12)
        result.setdefault("cfg", 1)
    elif "z-image" in combined or "zimage" in combined or "z_image" in combined:
        result["preset"] = "ZImageTurbo"
        result.setdefault("scheduler", "simple")
        result.setdefault("sampler", "res_multistep")
    elif "ltx" in combined:
        result["preset"] = "LTX"
        result.setdefault("scheduler", "quadratic")
        result.setdefault("sampler", "euler_cfg_pp")
    elif "wan" in combined:
        result["preset"] = "Wan"
    elif "anima" in combined:
        result["preset"] = "Anima"
    elif "sdxl" in combined:
        result["preset"] = "XL"
    elif "flux" in combined:
        result["preset"] = "Flux"
    elif "qwen" in combined:
        result["preset"] = "Qwen"
    elif "lumina" in combined:
        result["preset"] = "Lumina"
    if result.get("preset") == "LTX" and result.get("frames") and result.get("fps") and "seconds" not in result:
        result["seconds"] = round(float(result["frames"]) / max(float(result["fps"]), 1.0), 2)
    return result


def _known_ui_widget_order(node_type: str) -> list[str]:
    lower = node_type.lower()
    if "qwen" in lower and "textencode" in lower:
        return ["prompt"]
    if lower == "ltxdirector":
        return [
            "global_prompt",
            "duration_frames",
            "duration_seconds",
            "timeline_data",
            "use_custom_audio",
            "local_prompts",
            "segment_lengths",
            "epsilon",
            "guide_strength",
            "frame_rate",
            "display_mode",
            "custom_width",
            "custom_height",
            "resize_method",
            "divisible_by",
            "img_compression",
        ]
    if lower == "modelsamplingauraflow":
        return ["shift"]
    if lower == "textencodezimageomni":
        return ["prompt", "auto_resize_images"]
    if "textencode" in lower or "cliptext" in lower:
        return ["text"]
    if "ksamplerselect" in lower:
        return ["sampler_name"]
    if "ksampler" in lower:
        return ["seed", "control_after_generate", "steps", "cfg", "sampler_name", "scheduler", "denoise"]
    if "cfgguider" in lower:
        return ["cfg"]
    if lower == "loadimage":
        return ["image", "upload"]
    if lower == "trellis2loadimagewithtransparency":
        return ["image", "upload"]
    if lower == "trellis2loadmodel":
        return [
            "modelname",
            "backend",
            "device",
            "low_vram",
            "keep_models_loaded",
            "conv_backend",
            "sparse_backend",
            "use_reconviagen",
        ]
    if lower == "trellis2preprocessimage":
        return ["padding", "remove_background", "max_size"]
    if lower == "trellis2sparsemultiviewgenerator":
        return [
            "seed",
            "control_after_generate",
            "sparse_structure_steps",
            "sparse_structure_guidance_strength",
            "sparse_structure_guidance_rescale",
            "sparse_structure_rescale_t",
            "sparse_structure_sampler",
            "sparse_structure_resolution",
            "sparse_structure_guidance_interval_start",
            "sparse_structure_guidance_interval_end",
            "fill_holes",
            "hole_iterations",
            "verbose",
            "dino_lock",
            "dino_substeps",
            "hole_fill_algorithm",
            "dino_foundation_cap",
            "keep_only_shell",
            "front_axis",
            "blend_temperature",
        ]
    if lower == "trellis2shapemultiviewgenerator":
        return [
            "resolution",
            "shape_steps",
            "shape_guidance_strength",
            "shape_guidance_rescale",
            "shape_rescale_t",
            "shape_sampler",
            "shape_guidance_interval_start",
            "shape_guidance_interval_end",
            "verbose",
            "dino_lock",
            "dino_substeps",
            "dino_foundation_cap",
            "front_axis",
            "blend_temperature",
        ]
    if lower == "trellis2shapecascademultiviewgenerator":
        return [
            "seed",
            "to_resolution",
            "sparse_structure_resolution",
            "max_num_tokens",
            "shape_steps",
            "shape_guidance_strength",
            "shape_guidance_rescale",
            "shape_rescale_t",
            "shape_sampler",
            "shape_guidance_interval_start",
            "shape_guidance_interval_end",
            "verbose",
            "dino_lock",
            "dino_substeps",
            "dino_foundation_cap",
            "front_axis",
            "blend_temperature",
        ]
    if lower == "trellis2texslatmultiviewgenerator":
        return ["resolution", "texture_steps", "texture_guidance_strength", "texture_guidance_rescale", "texture_rescale_t", "texture_sampler", "temperature", "scale"]
    if lower in {"primitiveint", "primitivestring"}:
        return ["value", "control_after_generate"]
    if lower == "loadvideo":
        return ["video", "output_mode"]
    if lower in {"vhs_loadvideo", "loadvideoui"}:
        return [
            "video",
            "force_rate",
            "custom_width",
            "custom_height",
            "frame_load_cap",
            "skip_first_frames",
            "select_every_nth",
        ]
    if "lora" in lower:
        return ["lora_name", "strength_model", "strength_clip", "video", "video_to_audio", "audio", "audio_to_video", "other"]
    if "vaeloader" in lower:
        return ["vae_name"]
    if "cliploader" in lower or "textencoderloader" in lower:
        return ["clip_name", "type", "device"]
    if "ltxvconditioning" in lower:
        return ["frame_rate"]
    if lower == "randomnoise":
        return ["noise_seed", "control_after_generate"]
    if lower in {"easy int", "easy float", "intconstant", "floatconstant"}:
        return ["value"]
    if "ltxvemptylatentaudio" in lower:
        return ["frames_number", "frame_rate", "batch_size"]
    if "emptylatent" in lower or "latentvideo" in lower:
        return ["width", "height", "length", "batch_size"]
    if lower == "emptyimage":
        return ["width", "height", "batch_size", "color"]
    if "rifeinterpolation" in lower:
        return ["source_fps", "target_fps", "scale", "model_name", "batch_size", "use_fp16"]
    if "duallcliploader" in lower:
        return ["clip_name1", "clip_name2", "type", "device"]
    if lower in {"unetloader", "diffusionmodelloader", "diffusionmodelloaderkj"}:
        return ["unet_name", "weight_dtype"]
    return []


def _ui_workflow_graph(data: dict[str, Any]) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    for index, node in enumerate(data.get("nodes", []) or []):
        node_id = str(node.get("id", index))
        class_type = str(node.get("type") or node.get("class_type") or "Unknown")
        title = str(node.get("title") or node.get("properties", {}).get("Node name for S&R") or class_type)
        x, y = _pair(node.get("pos"), (60 + (index % 5) * 260, 80 + (index // 5) * 160))
        width, height = _pair(node.get("size"), (220, 120))
        inputs = [str(item.get("name")) for item in node.get("inputs", []) or [] if item.get("name")]
        outputs = [str(item.get("name")) for item in node.get("outputs", []) or [] if item.get("name")]
        widgets = _widget_values(node.get("widgets_values"))
        flags = node.get("flags") or {}
        note_text = ""
        if "note" in f"{class_type} {title}".lower() and widgets:
            note_text = widgets[0].get("value", "")
        nodes.append(
            {
                "id": node_id,
                "class_type": class_type,
                "title": title,
                "x": int(x),
                "y": int(y),
                "width": max(190, min(int(width or 220), 340)),
                "height": max(58, min(int(height or 120), 260)),
                "inputs": inputs[:10],
                "outputs": outputs[:8],
                "widgets": widgets[:8],
                "color": _safe_hex_color(str(node.get("color") or "")),
                "bgcolor": _safe_hex_color(str(node.get("bgcolor") or "")),
                "collapsed": bool(flags.get("collapsed")),
                "pinned": bool(flags.get("pinned")),
                "bypassed": int(node.get("mode") or 0) == 4 or bool(flags.get("bypassed")),
                "mode": int(node.get("mode") or 0),
                "note": note_text,
            }
        )

    links: list[dict[str, Any]] = []
    for link in data.get("links", []) or []:
        if isinstance(link, list) and len(link) >= 5:
            links.append(
                {
                    "from_node": str(link[1]),
                    "from_slot": int(link[2] or 0),
                    "to_node": str(link[3]),
                    "to_slot": int(link[4] or 0),
                    "type": str(link[5]) if len(link) > 5 else "",
                }
            )

    groups = _ui_groups(data)
    return _with_graph_bounds(_resolve_overlaps(nodes), links, groups)


def _api_workflow_graph(data: dict[str, Any]) -> dict[str, Any]:
    node_ids = [
        str(key)
        for key, value in data.items()
        if isinstance(value, dict) and "class_type" in value
    ]
    links: list[dict[str, Any]] = []
    parents_by_node: dict[str, list[str]] = {node_id: [] for node_id in node_ids}
    for node_id, node in data.items():
        if not isinstance(node, dict):
            continue
        for slot, value in enumerate((node.get("inputs") or {}).values()):
            if isinstance(value, list) and value:
                parent_id = str(value[0])
                links.append({"from_node": parent_id, "from_slot": int(value[1] or 0), "to_node": str(node_id), "to_slot": slot, "type": ""})
                parents_by_node.setdefault(str(node_id), []).append(parent_id)

    depth: dict[str, int] = {node_id: 0 for node_id in node_ids}
    for _ in range(len(node_ids)):
        changed = False
        for node_id, parents in parents_by_node.items():
            if parents:
                next_depth = max(depth.get(parent, 0) + 1 for parent in parents)
                if next_depth > depth.get(node_id, 0):
                    depth[node_id] = next_depth
                    changed = True
        if not changed:
            break

    rows_by_depth: dict[int, int] = {}
    nodes: list[dict[str, Any]] = []
    for node_id in sorted(node_ids, key=_node_sort_key):
        node = data.get(node_id, {})
        class_type = str(node.get("class_type", "Unknown")) if isinstance(node, dict) else "Unknown"
        title = str(node.get("_meta", {}).get("title") or class_type) if isinstance(node, dict) else class_type
        current_depth = depth.get(node_id, 0)
        row = rows_by_depth.get(current_depth, 0)
        rows_by_depth[current_depth] = row + 1
        inputs = list((node.get("inputs") or {}).keys()) if isinstance(node, dict) else []
        widgets = [
            {"name": str(key), "value": _short_value(value)}
            for key, value in (node.get("inputs") or {}).items()
            if not isinstance(value, list)
        ]
        nodes.append(
            {
                "id": node_id,
                "class_type": class_type,
                "title": title,
                "x": 60 + current_depth * 270,
                "y": 80 + row * 150,
                "width": 230,
                "height": 118,
                "inputs": [str(item) for item in inputs[:10]],
                "outputs": [],
                "widgets": widgets[:8],
            }
        )
    return _with_graph_bounds(_resolve_overlaps(nodes), links)


def _organize_nodes(nodes: list[dict[str, Any]], links: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not nodes:
        return nodes
    node_by_id = {str(node["id"]): node for node in nodes}
    parents: dict[str, list[str]] = {node_id: [] for node_id in node_by_id}
    for link in links:
        from_id = str(link.get("from_node"))
        to_id = str(link.get("to_node"))
        if from_id in node_by_id and to_id in node_by_id:
            parents.setdefault(to_id, []).append(from_id)

    depth = {node_id: 0 for node_id in node_by_id}
    for _ in range(len(nodes)):
        changed = False
        for node_id, parent_ids in parents.items():
            if not parent_ids:
                continue
            next_depth = max(depth.get(parent_id, 0) + 1 for parent_id in parent_ids)
            if next_depth > depth[node_id]:
                depth[node_id] = next_depth
                changed = True
        if not changed:
            break

    rows: dict[int, int] = {}
    for node in sorted(nodes, key=lambda item: (depth.get(str(item["id"]), 0), int(item.get("y", 0)), int(item.get("x", 0)))):
        d = depth.get(str(node["id"]), 0)
        row = rows.get(d, 0)
        rows[d] = row + 1
        node["x"] = 60 + d * 300
        node["y"] = 80 + row * 165
    return nodes


def _resolve_overlaps(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    placed: list[dict[str, Any]] = []
    for node in sorted(nodes, key=lambda item: (int(item.get("y", 0)), int(item.get("x", 0)), str(item.get("id")))):
        node["width"] = max(190, min(int(node.get("width") or 220), 420))
        node["height"] = max(90, min(int(node.get("height") or 120), 360))
        guard = 0
        while any(_rects_overlap(node, other) for other in placed) and guard < 120:
            node["y"] = int(node.get("y") or 0) + 32
            guard += 1
        placed.append(node)
    return nodes


def _rects_overlap(a: dict[str, Any], b: dict[str, Any]) -> bool:
    gap = 18
    ax, ay = int(a.get("x", 0)), int(a.get("y", 0))
    aw, ah = int(a.get("width", 220)), int(a.get("height", 120))
    bx, by = int(b.get("x", 0)), int(b.get("y", 0))
    bw, bh = int(b.get("width", 220)), int(b.get("height", 120))
    return ax < bx + bw + gap and ax + aw + gap > bx and ay < by + bh + gap and ay + ah + gap > by


def _ui_groups(data: dict[str, Any]) -> list[dict[str, Any]]:
    groups: list[dict[str, Any]] = []
    for group in data.get("groups", []) or []:
        if not isinstance(group, dict):
            continue
        x, y, width, height = _quad(group.get("bounding"), (60, 60, 320, 220))
        groups.append(
            {
                "id": str(group.get("id") or len(groups)),
                "title": str(group.get("title") or "Group"),
                "x": int(x),
                "y": int(y),
                "width": max(160, int(width)),
                "height": max(100, int(height)),
                "color": _safe_hex_color(str(group.get("color") or "#3f789e")),
            }
        )
    return groups


def _with_graph_bounds(nodes: list[dict[str, Any]], links: list[dict[str, Any]], groups: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    groups = groups or []
    min_x = min([int(node["x"]) for node in nodes] + [int(group["x"]) for group in groups] or [60])
    min_y = min([int(node["y"]) for node in nodes] + [int(group["y"]) for group in groups] or [60])
    shift_x = 60 - min_x if min_x < 40 else 0
    shift_y = 60 - min_y if min_y < 40 else 0
    if shift_x or shift_y:
        for node in nodes:
            node["x"] = int(node["x"]) + shift_x
            node["y"] = int(node["y"]) + shift_y
        for group in groups:
            group["x"] = int(group["x"]) + shift_x
            group["y"] = int(group["y"]) + shift_y
    max_x = max([int(node["x"]) + int(node["width"]) for node in nodes] + [int(group["x"]) + int(group["width"]) for group in groups] or [1100])
    max_y = max([int(node["y"]) + int(node["height"]) for node in nodes] + [int(group["y"]) + int(group["height"]) for group in groups] or [620])
    return {
        "nodes": nodes,
        "links": links,
        "groups": groups,
        "width": max(1300, max_x + 220),
        "height": max(720, max_y + 180),
    }


def _pair(value: Any, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, dict):
        values = list(value.values())
    else:
        values = value
    if isinstance(values, (list, tuple)) and len(values) >= 2:
        try:
            return float(values[0]), float(values[1])
        except (TypeError, ValueError):
            pass
    return fallback


def _quad(value: Any, fallback: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    if isinstance(value, dict):
        values = list(value.values())
    else:
        values = value
    if isinstance(values, (list, tuple)) and len(values) >= 4:
        try:
            return float(values[0]), float(values[1]), float(values[2]), float(values[3])
        except (TypeError, ValueError):
            pass
    return fallback


def _safe_hex_color(value: str) -> str:
    value = value.strip()
    if re.fullmatch(r"#[0-9a-fA-F]{3,8}", value):
        return value
    return ""


def _widget_values(value: Any) -> list[dict[str, str]]:
    if isinstance(value, dict):
        return [{"name": str(key), "value": _short_value(item)} for key, item in value.items()]
    if isinstance(value, list):
        return [{"name": f"widget {index + 1}", "value": _short_value(item)} for index, item in enumerate(value)]
    return []


def _short_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    return text if len(text) <= 64 else f"{text[:61]}..."


def _node_sort_key(value: str) -> tuple[int, str]:
    return (int(value), value) if value.isdigit() else (10**9, value)


def _is_ui_helper_node(class_type: str) -> bool:
    if class_type in UI_HELPER_NODE_TYPES:
        return True
    return bool(re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", class_type))


def _ui_link_tuple(link: Any) -> tuple[int, str, int, str, int, str] | None:
    if isinstance(link, list) and len(link) >= 6:
        return int(link[0]), str(link[1]), int(link[2]), str(link[3]), int(link[4]), str(link[5])
    if isinstance(link, dict):
        link_id = int(link.get("id") or 0)
        origin_id = str(link.get("origin_id"))
        target_id = str(link.get("target_id"))
        if not link_id or not origin_id or not target_id:
            return None
        return (
            link_id,
            origin_id,
            int(link.get("origin_slot") or 0),
            target_id,
            int(link.get("target_slot") or 0),
            str(link.get("type") or "*"),
        )
    return None


def _expand_ui_subgraphs(data: dict[str, Any]) -> dict[str, Any]:
    subgraphs = {
        str(subgraph.get("id")): subgraph
        for subgraph in data.get("definitions", {}).get("subgraphs", []) or []
        if isinstance(subgraph, dict) and subgraph.get("id")
    }
    if not subgraphs:
        return data

    wrapper_nodes = [
        node
        for node in data.get("nodes", []) or []
        if isinstance(node, dict) and str(node.get("type") or node.get("class_type") or "") in subgraphs
    ]
    if not wrapper_nodes:
        return data

    expanded = deepcopy(data)
    top_links = [
        parsed
        for parsed in (_ui_link_tuple(link) for link in expanded.get("links", []) or [])
        if parsed is not None
    ]
    wrapper_ids = {str(node.get("id")) for node in wrapper_nodes}
    incoming: dict[tuple[str, int], list[tuple[int, str, int, str, int, str]]] = {}
    outgoing: dict[tuple[str, int], list[tuple[int, str, int, str, int, str]]] = {}
    for link in top_links:
        _, origin_id, origin_slot, target_id, target_slot, _ = link
        if target_id in wrapper_ids:
            incoming.setdefault((target_id, target_slot), []).append(link)
        if origin_id in wrapper_ids:
            outgoing.setdefault((origin_id, origin_slot), []).append(link)

    nodes: list[dict[str, Any]] = []
    link_specs: list[tuple[str, int, str, int, str]] = []
    for node in expanded.get("nodes", []) or []:
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id"))
        class_type = str(node.get("type") or node.get("class_type") or "")
        subgraph = subgraphs.get(class_type)
        if not subgraph:
            nodes.append(node)
            continue

        nodes.extend(deepcopy(subgraph.get("nodes", []) or []))
        for link in top_links:
            _, origin_id, origin_slot, target_id, target_slot, link_type = link
            if origin_id != node_id and target_id != node_id:
                link_specs.append((origin_id, origin_slot, target_id, target_slot, link_type))
        for raw_link in subgraph.get("links", []) or []:
            parsed = _ui_link_tuple(raw_link)
            if not parsed:
                continue
            _, origin_id, origin_slot, target_id, target_slot, link_type = parsed
            if origin_id == "-10":
                for source in incoming.get((node_id, origin_slot), []):
                    _, top_origin_id, top_origin_slot, _, _, top_type = source
                    link_specs.append((top_origin_id, top_origin_slot, target_id, target_slot, top_type or link_type))
            elif target_id == "-20":
                for target in outgoing.get((node_id, target_slot), []):
                    _, _, _, top_target_id, top_target_slot, top_type = target
                    link_specs.append((origin_id, origin_slot, top_target_id, top_target_slot, top_type or link_type))
            else:
                link_specs.append((origin_id, origin_slot, target_id, target_slot, link_type))

    # Deduplicate because each wrapper replacement already emits untouched top-level links.
    seen: set[tuple[str, int, str, int, str]] = set()
    deduped: list[tuple[str, int, str, int, str]] = []
    for spec in link_specs:
        if spec in seen:
            continue
        seen.add(spec)
        deduped.append(spec)

    node_by_id = {str(node.get("id")): node for node in nodes if isinstance(node, dict)}
    for node in node_by_id.values():
        for input_info in node.get("inputs", []) or []:
            if isinstance(input_info, dict):
                input_info["link"] = None
        for output_info in node.get("outputs", []) or []:
            if isinstance(output_info, dict):
                output_info["links"] = []

    links: list[list[Any]] = []
    for index, (origin_id, origin_slot, target_id, target_slot, link_type) in enumerate(deduped, start=1):
        link_id = index
        links.append([link_id, origin_id, origin_slot, target_id, target_slot, link_type])
        target = node_by_id.get(str(target_id))
        target_inputs = target.get("inputs", []) if isinstance(target, dict) else []
        if 0 <= target_slot < len(target_inputs) and isinstance(target_inputs[target_slot], dict):
            target_inputs[target_slot]["link"] = link_id
        origin = node_by_id.get(str(origin_id))
        origin_outputs = origin.get("outputs", []) if isinstance(origin, dict) else []
        if 0 <= origin_slot < len(origin_outputs) and isinstance(origin_outputs[origin_slot], dict):
            origin_outputs[origin_slot].setdefault("links", []).append(link_id)

    expanded["nodes"] = nodes
    expanded["links"] = links
    expanded["last_link_id"] = len(links)
    expanded["last_node_id"] = max((int(node.get("id")) for node in nodes if isinstance(node, dict) and str(node.get("id")).lstrip("-").isdigit()), default=0)
    if any(isinstance(node, dict) and str(node.get("type") or node.get("class_type") or "") in subgraphs for node in nodes):
        return _expand_ui_subgraphs(expanded)
    return expanded


def convert_ui_to_api(data: dict[str, Any], object_info: dict[str, Any]) -> dict[str, Any]:
    data = _expand_ui_subgraphs(data)
    links: dict[int, tuple[str, int]] = {}
    for link in data.get("links", []):
        if isinstance(link, list) and len(link) >= 6:
            link_id, origin_id, origin_slot = int(link[0]), str(link[1]), int(link[2])
            links[link_id] = (origin_id, origin_slot)

    api: dict[str, Any] = {}
    for node in data.get("nodes", []):
        node_id = str(node.get("id"))
        class_type = node.get("type") or node.get("class_type")
        if not node_id or not class_type:
            continue
        if object_info and class_type not in object_info and _is_ui_helper_node(str(class_type)):
            continue
        inputs: dict[str, Any] = {}
        linked_widget_names: set[str] = set()
        for input_info in node.get("inputs", []) or []:
            name = str(input_info.get("name") or "")
            link = input_info.get("link")
            widget = input_info.get("widget")
            if name and isinstance(widget, dict) and widget.get("name"):
                linked_widget_names.add(str(widget.get("name")))
            if name and link is not None and int(link) in links:
                origin_id, origin_slot = links[int(link)]
                inputs[name] = [origin_id, origin_slot]

        widget_values = node.get("widgets_values", [])
        named_widget_values = node.get("properties", {}).get("nexus_widget_values")
        if isinstance(named_widget_values, dict):
            for key, value in named_widget_values.items():
                inputs.setdefault(key, value)
        elif isinstance(widget_values, dict):
            for key, value in widget_values.items():
                inputs.setdefault(key, value)
        else:
            widget_names = _widget_input_names(class_type, object_info)
            value_iter = iter(widget_values if isinstance(widget_values, list) else [])
            for name in widget_names:
                if name in inputs and name not in linked_widget_names:
                    continue
                value = None
                has_value = False
                try:
                    value = next(value_iter)
                    has_value = True
                except StopIteration:
                    break
                if name in inputs:
                    if name in linked_widget_names:
                        continue
                    continue
                if has_value:
                    inputs[name] = value

        _patch_dynamic_ui_widget_values(str(class_type), inputs, widget_values)

        if object_info and class_type in object_info:
            node_inputs = object_info.get(class_type, {}).get("input", {})
            allowed_inputs = _allowed_comfy_input_names(node_inputs)
            if allowed_inputs:
                inputs = {key: value for key, value in inputs.items() if key in allowed_inputs}

        api[node_id] = {
            "class_type": class_type,
            "inputs": inputs,
        }
        title = node.get("title") or node.get("properties", {}).get("Node name for S&R")
        if title:
            api[node_id]["_meta"] = {"title": title}
    return api


def _allowed_comfy_input_names(node_inputs: dict[str, Any]) -> set[str]:
    allowed_inputs: set[str] = set()
    for group in ("required", "optional", "hidden"):
        values = node_inputs.get(group, {})
        if not isinstance(values, dict):
            continue
        allowed_inputs.update(values.keys())
        for parent_name, spec in values.items():
            if not isinstance(spec, list) or len(spec) < 2 or not isinstance(spec[1], dict):
                continue
            meta = spec[1]
            options = meta.get("options")
            if isinstance(options, list):
                for option in options:
                    if not isinstance(option, dict):
                        continue
                    option_inputs = option.get("inputs", {})
                    if not isinstance(option_inputs, dict):
                        continue
                    for option_group in ("required", "optional", "hidden"):
                        option_values = option_inputs.get(option_group, {})
                        if isinstance(option_values, dict):
                            allowed_inputs.update(option_values.keys())
                            allowed_inputs.update(f"{parent_name}.{key}" for key in option_values.keys())
            template = meta.get("template")
            if isinstance(template, dict) and isinstance(template.get("names"), list):
                allowed_inputs.update(str(name) for name in template["names"])
                allowed_inputs.update(f"{parent_name}.{name}" for name in template["names"])
    return allowed_inputs


def _patch_dynamic_ui_widget_values(class_type: str, inputs: dict[str, Any], widget_values: Any) -> None:
    if not isinstance(widget_values, list) or not widget_values:
        return
    lower = class_type.lower()
    if lower == "createcamerainfo":
        mode_value = str(widget_values[0] if len(widget_values) >= 1 else inputs.get("mode") or "orbit")
        if mode_value == "orbit":
            inputs.setdefault("mode", "orbit")
            inputs.setdefault("yaw", widget_values[1] if len(widget_values) >= 2 else 35)
            inputs.setdefault("pitch", widget_values[2] if len(widget_values) >= 3 else 30)
            inputs.setdefault("distance", widget_values[3] if len(widget_values) >= 4 else 4)
            offset = 4
        else:
            inputs.setdefault("mode", "orbit")
            inputs.setdefault("yaw", 35)
            inputs.setdefault("pitch", 30)
            inputs.setdefault("distance", 4)
            offset = 4
        names = ["target_x", "target_y", "target_z", "roll", "fov", "zoom", "camera_type"]
        for index, name in enumerate(names, start=offset):
            if index < len(widget_values):
                inputs.setdefault(name, widget_values[index])
        return
    if lower == "resizeimagemasknode":
        resize_type = str(widget_values[0] or "")
        inputs["resize_type"] = resize_type
        if resize_type == "scale by multiplier" and len(widget_values) >= 3:
            inputs["resize_type.multiplier"] = widget_values[1]
            inputs["scale_method"] = widget_values[2]
        elif resize_type == "scale to multiple" and len(widget_values) >= 3:
            inputs["resize_type.multiple"] = widget_values[1]
            inputs["scale_method"] = widget_values[2]
        elif resize_type == "match size" and len(widget_values) >= 3:
            inputs["resize_type.crop"] = widget_values[1]
            inputs["scale_method"] = widget_values[2]
        elif resize_type == "scale dimensions" and len(widget_values) >= 5:
            inputs["resize_type.width"] = widget_values[1]
            inputs["resize_type.height"] = widget_values[2]
            inputs["resize_type.crop"] = widget_values[3]
            inputs["scale_method"] = widget_values[4]
        elif len(widget_values) >= 2:
            # Other resize modes expose one numeric option followed by scale_method.
            mode_key = {
                "scale longer dimension": "longer_size",
                "scale shorter dimension": "shorter_size",
                "scale width": "width",
                "scale height": "height",
            }.get(resize_type)
            if mode_key:
                inputs[f"resize_type.{mode_key}"] = widget_values[1]
                if len(widget_values) >= 3:
                    inputs["scale_method"] = widget_values[2]


def _widget_input_names(class_type: str, object_info: dict[str, Any]) -> list[str]:
    lower = class_type.lower()
    if lower == "comfymathexpression":
        return ["expression"]
    if lower == "resizeimagemasknode":
        return []
    known = _known_ui_widget_order(class_type)
    if lower in {"trellis2sparsemultiviewgenerator"} and known:
        return known
    info = object_info.get(class_type, {}).get("input", {}) if object_info else {}
    names: list[str] = []
    for group in ["required", "optional"]:
        values = info.get(group, {})
        if isinstance(values, dict):
            names.extend(values.keys())
    for name in known:
        if name not in names:
            names.append(name)
    return names


def _append_inpaint_mask(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    reference_node_id: str,
    vae_ref: list[Any],
    sampler_node_id: str,
    mask_image_name: str | None,
    mask_ref_override: list[Any] | None = None,
    start_id: int = 80,
    decoded_image_ref: list[Any] | None = None,
    save_node_id: str | None = None,
    available_nodes: set[str] | None = None,
) -> bool:
    if not (mask_image_name or mask_ref_override) or not _uses_inpaint_mask_mode(request):
        return False
    mask_loader_id = str(start_id)
    mask_to_mask_id = str(start_id + 1)
    encode_id = str(start_id + 2)
    reference_resize_id = str(start_id + 3)
    mask_resize_id = str(start_id + 4)
    next_id = start_id + 5
    grow_mask_by = max(0, min(64, int(request.img2img.mask_blur or 0)))
    available_nodes = set(available_nodes or ())
    workflow[reference_resize_id] = _image_scale_node([reference_node_id, 0], request.width, request.height)
    workflow[reference_resize_id]["_meta"]["title"] = "Resize Inpaint Reference To Side Menu"
    if mask_ref_override:
        mask_ref = mask_ref_override
        next_id = start_id + 4
    else:
        workflow[mask_loader_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image_name},
            "_meta": {"title": "Inpaint Mask"},
        }
        workflow[mask_resize_id] = _image_scale_node([mask_loader_id, 0], request.width, request.height, method="nearest-exact")
        workflow[mask_resize_id]["_meta"]["title"] = "Resize Inpaint Mask To Side Menu"
        workflow[mask_to_mask_id] = {
            "class_type": "ImageToMask",
            "inputs": {"image": [mask_resize_id, 0], "channel": "red"},
            "_meta": {"title": "Mask Channel"},
        }
        mask_ref = [mask_to_mask_id, 0]
    if _uses_outpaint_extend_mode(request) and "GrowMaskWithBlur" in available_nodes:
        grow_mask_id = str(next_id)
        next_id += 1
        workflow[grow_mask_id] = {
            "class_type": "GrowMaskWithBlur",
            "inputs": {
                "mask": mask_ref,
                "expand": 20,
                "incremental_expandrate": 0.0,
                "tapered_corners": True,
                "flip_input": False,
                "blur_radius": 4.0,
                "lerp_alpha": 1.0,
                "decay_factor": 1.0,
                "fill_holes": False,
            },
            "_meta": {"title": "Grow Outpaint Mask For Seamless Blend"},
        }
        mask_ref = [grow_mask_id, 0]
    workflow[encode_id] = {
        "class_type": "VAEEncodeForInpaint",
        "inputs": {
            "pixels": [reference_resize_id, 0],
            "vae": vae_ref,
            "mask": mask_ref,
            "grow_mask_by": grow_mask_by,
        },
        "_meta": {"title": "Encode Reference For Inpaint"},
    }
    sampler = workflow.get(sampler_node_id, {})
    sampler_inputs = sampler.setdefault("inputs", {})
    sampler_inputs["latent_image"] = [encode_id, 0]
    if decoded_image_ref and save_node_id:
        _append_masked_output_composite(
            workflow,
            request=request,
            destination_image_ref=[reference_resize_id, 0],
            source_image_ref=decoded_image_ref,
            mask_ref=mask_ref,
            save_node_id=save_node_id,
            start_id=next_id,
            available_nodes=available_nodes,
            prefer_request_composite_mask=mask_ref_override is None,
        )
    return True


def _append_masked_output_composite(
    workflow: dict[str, Any],
    *,
    request: GenerateRequest | None = None,
    destination_image_ref: list[Any],
    source_image_ref: list[Any],
    mask_ref: list[Any],
    save_node_id: str,
    start_id: int,
    available_nodes: set[str] | None = None,
    prefer_request_composite_mask: bool = True,
) -> list[Any] | None:
    save_node = workflow.get(save_node_id)
    if not isinstance(save_node, dict):
        return None
    composite_id = str(start_id)
    while composite_id in workflow:
        start_id += 1
        composite_id = str(start_id)
    composite_mask_ref = mask_ref
    composite_mask_image_name = ""
    if prefer_request_composite_mask and request is not None and _uses_outpaint_extend_mode(request):
        composite_mask_image_name = str(getattr(request.img2img, "composite_mask_image", "") or "").strip()
    if composite_mask_image_name:
        load_mask_id = composite_id
        start_id += 1
        resize_mask_id = str(start_id)
        while resize_mask_id in workflow:
            start_id += 1
            resize_mask_id = str(start_id)
        start_id += 1
        mask_to_mask_id = str(start_id)
        while mask_to_mask_id in workflow:
            start_id += 1
            mask_to_mask_id = str(start_id)
        start_id += 1
        composite_id = str(start_id)
        while composite_id in workflow:
            start_id += 1
            composite_id = str(start_id)
        width = max(1, int(getattr(request, "width", 0) or 0))
        height = max(1, int(getattr(request, "height", 0) or 0))
        workflow[load_mask_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": composite_mask_image_name},
            "_meta": {"title": "Extend Composite Mask"},
        }
        workflow[resize_mask_id] = _image_scale_node([load_mask_id, 0], width, height, method="nearest-exact")
        workflow[resize_mask_id]["_meta"]["title"] = "Resize Extend Composite Mask"
        workflow[mask_to_mask_id] = {
            "class_type": "ImageToMask",
            "inputs": {"image": [resize_mask_id, 0], "channel": "red"},
            "_meta": {"title": "Extend Composite Mask Channel"},
        }
        composite_mask_ref = [mask_to_mask_id, 0]
    if request is not None:
        try:
            composite_blur = max(0, min(64, int(getattr(request.img2img, "mask_blur", 0) or 0)))
        except (TypeError, ValueError):
            composite_blur = 0
        if composite_blur > 0 and "GrowMaskWithBlur" in set(available_nodes or ()):
            blur_id = composite_id
            workflow[blur_id] = {
                "class_type": "GrowMaskWithBlur",
                "inputs": {
                    "mask": composite_mask_ref,
                    "expand": 0,
                    "incremental_expandrate": 0.0,
                    "tapered_corners": True,
                    "flip_input": False,
                    "blur_radius": float(composite_blur),
                    "lerp_alpha": 1.0,
                    "decay_factor": 1.0,
                    "fill_holes": False,
                },
                "_meta": {"title": "Soften Inpaint Composite Mask"},
            }
            composite_mask_ref = [blur_id, 0]
            start_id += 1
            composite_id = str(start_id)
            while composite_id in workflow:
                start_id += 1
                composite_id = str(start_id)
    workflow[composite_id] = {
        "class_type": "ImageCompositeMasked",
        "inputs": {
            "destination": destination_image_ref,
            "source": source_image_ref,
            "x": 0,
            "y": 0,
            "resize_source": False,
            "mask": composite_mask_ref,
        },
        "_meta": {"title": "Composite Generated Mask Over Preserved Source"},
    }
    output_ref: list[Any] = [composite_id, 0]
    if request is not None:
        width = max(1, int(getattr(request, "width", 0) or 0))
        height = max(1, int(getattr(request, "height", 0) or 0))
        if width and height:
            scale_id = str(start_id + 1)
            while scale_id in workflow:
                start_id += 1
                scale_id = str(start_id + 1)
            workflow[scale_id] = _image_scale_node(output_ref, width, height)
            workflow[scale_id]["_meta"]["title"] = "Resize Composite To Requested Output"
            output_ref = [scale_id, 0]
    save_node.setdefault("inputs", {})["images"] = output_ref
    return output_ref


def _append_qwen_inpaint_noise_mask(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    base_latent_ref: list[Any],
    sampler_node_id: str,
    mask_image_name: str | None,
    mask_ref_override: list[Any] | None = None,
    start_id: int = 80,
    base_image_ref: list[Any] | None = None,
    decoded_image_ref: list[Any] | None = None,
    save_node_id: str | None = None,
    available_nodes: set[str] | None = None,
) -> bool:
    if not (mask_image_name or mask_ref_override) or not _uses_inpaint_mask_mode(request):
        return False
    mask_loader_id = str(start_id)
    mask_to_mask_id = str(start_id + 1)
    set_mask_id = str(start_id + 2)
    mask_resize_id = str(start_id + 4)
    next_id = start_id + 5
    available_nodes = set(available_nodes or ())
    if mask_ref_override:
        mask_ref = mask_ref_override
        next_id = start_id + 3
    else:
        workflow[mask_loader_id] = {
            "class_type": "LoadImage",
            "inputs": {"image": mask_image_name},
            "_meta": {"title": "Inpaint Mask"},
        }
        workflow[mask_resize_id] = _image_scale_node([mask_loader_id, 0], request.width, request.height, method="nearest-exact")
        workflow[mask_resize_id]["_meta"]["title"] = "Resize Inpaint Mask To Side Menu"
        workflow[mask_to_mask_id] = {
            "class_type": "ImageToMask",
            "inputs": {"image": [mask_resize_id, 0], "channel": "red"},
            "_meta": {"title": "Mask Channel"},
        }
        mask_ref = [mask_to_mask_id, 0]
    if _uses_outpaint_extend_mode(request) and "GrowMaskWithBlur" in available_nodes:
        grow_mask_id = str(next_id)
        next_id += 1
        workflow[grow_mask_id] = {
            "class_type": "GrowMaskWithBlur",
            "inputs": {
                "mask": mask_ref,
                "expand": 20,
                "incremental_expandrate": 0.0,
                "tapered_corners": True,
                "flip_input": False,
                "blur_radius": 4.0,
                "lerp_alpha": 1.0,
                "decay_factor": 1.0,
                "fill_holes": False,
            },
            "_meta": {"title": "Grow QWEN Outpaint Mask For Seamless Blend"},
        }
        mask_ref = [grow_mask_id, 0]
    workflow[set_mask_id] = {
        "class_type": "SetLatentNoiseMask",
        "inputs": {"samples": base_latent_ref, "mask": mask_ref},
        "_meta": {"title": "Apply QWEN Inpaint Noise Mask"},
    }
    sampler = workflow.get(sampler_node_id, {})
    sampler_inputs = sampler.setdefault("inputs", {})
    sampler_inputs["latent_image"] = [set_mask_id, 0]
    if base_image_ref and decoded_image_ref and save_node_id:
        _append_masked_output_composite(
            workflow,
            request=request,
            destination_image_ref=base_image_ref,
            source_image_ref=decoded_image_ref,
            mask_ref=mask_ref,
            save_node_id=save_node_id,
            start_id=next_id,
            available_nodes=available_nodes,
            prefer_request_composite_mask=mask_ref_override is None,
        )
    return True


def _uses_inpaint_mask_mode(request: GenerateRequest) -> bool:
    mode = str(request.img2img.mode or "").lower()
    return "inpaint" in mode or "outpaint" in mode or "extend" in mode


def _uses_outpaint_extend_mode(request: GenerateRequest) -> bool:
    mode = str(request.img2img.mode or "").lower()
    return "outpaint" in mode or "extend" in mode


def _extend_pad_values(request: GenerateRequest) -> dict[str, int]:
    raw = getattr(request.img2img, "extend_pad", None)
    if not isinstance(raw, dict):
        return {"left": 0, "top": 0, "right": 0, "bottom": 0}
    return {
        side: max(0, int(float(raw.get(side) or 0)))
        for side in ("left", "top", "right", "bottom")
    }


def _has_extend_pad(request: GenerateRequest) -> bool:
    return any(_extend_pad_values(request).values())


def _uses_prepadded_outpaint_reference(request: GenerateRequest) -> bool:
    # Extend/Outpaint should follow the workflow-node contract: load the
    # original image, then let the backend pad node produce the outpaint mask.
    # Treating the frontend preview as a pre-padded conditioning image makes
    # Qwen/Flux preserve or zoom the temporary padding instead of painting it.
    return False


def _image_pad_for_outpaint_node(
    image_ref: list[Any],
    request: GenerateRequest,
    feathering: int = 0,
    available_nodes: set[str] | None = None,
) -> dict[str, Any]:
    pad = _extend_pad_values(request)
    available_nodes = set(available_nodes or ())
    if "ImagePadForOutpaint" in available_nodes or "AGSoft_Img_Pad_Adv" not in available_nodes:
        return {
            "class_type": "ImagePadForOutpaint",
            "inputs": {
                "image": image_ref,
                "left": pad["left"],
                "top": pad["top"],
                "right": pad["right"],
                "bottom": pad["bottom"],
                "feathering": max(0, int(feathering or 0)),
            },
            "_meta": {"title": "Pad Base Image For Extend"},
        }
    return {
        "class_type": "AGSoft_Img_Pad_Adv",
        "inputs": {
            "image": image_ref,
            "pad_left": pad["left"],
            "pad_top": pad["top"],
            "pad_right": pad["right"],
            "pad_bottom": pad["bottom"],
            "pad_mode": "constant",
            "background_color": "gray",
            "feathering": 0,
            "invert_mask": False,
            "target_width": 0,
            "target_height": 0,
            "keep_proportions": False,
            "resize_position": "center",
        },
        "_meta": {"title": "AGSoft Pad Base Image For Extend"},
    }


def _apply_outpaint_continuity_prompt(request: GenerateRequest) -> None:
    if request.activity != "img2img" or not _uses_outpaint_extend_mode(request):
        return
    prefix = (
        "Edit only the masked areas and keep every unmasked pixel identical. "
        "If a masked area is temporary extension padding, replace it completely with newly painted scene content. "
        "Continue the visible background, subject, and environment naturally with matching lighting, perspective, texture, color, scale, and camera lens. "
        "Create plausible new scene detail in the expanded area; do not copy, mirror, smear, or stretch the edge pixels. "
        "Make the expanded area part of the same environment with no visible seam, divider, vertical stripe, solid-color block, placeholder padding, border, dark overlay, blur, zoom, crop, stretch, or distortion. "
        "Do not zoom, crop, stretch, darken, blur, or alter the original image outside the mask."
    )
    negative_prefix = (
        "visible seam, divider, split line, border, dark overlay, black border, blurred transition, "
        "placeholder padding, solid color block, brown block, vertical stripe, copied edge pixels, "
        "duplicated curtain, mirrored edge, smeared edge, stretched image, distorted perspective, zoom, crop, changed subject, changed face, changed body, changed unmasked area"
    )

    def merge(text: str, existing: str) -> str:
        current = str(existing or "").strip()
        lead = text.split(".", 1)[0].lower()
        if lead and lead in current.lower():
            return current
        return f"{text} {current}".strip()

    request.prompt = merge(prefix, request.prompt)
    request.negative_prompt = merge(negative_prefix, request.negative_prompt)


def _inpaint_engine(request: GenerateRequest) -> str:
    raw = str(getattr(request.img2img, "inpaint_engine", "") or "").strip().lower()
    if raw in {"lanpaint", "lan paint", "lanpaint_ksampler"}:
        return "lanpaint"
    if raw in {"default", "standard", "vae", "vaeencode", "off", "none", "disabled"}:
        return "default"
    if raw in {"differential", "differential_diffusion", "differential diffusion", "diffdiff"}:
        return "differential"
    return "differential" if _bool_option(getattr(request.img2img, "differential_diffusion", True), True) else "default"


def ensure_inpaint_engine_route(
    api: dict[str, Any],
    request: GenerateRequest,
    *,
    available_nodes: set[str] | None = None,
) -> None:
    if request.activity != "img2img" or not _uses_inpaint_mask_mode(request):
        return
    sampler_id = _find_sampler_node_id(api)
    if not sampler_id:
        return
    sampler = api.get(sampler_id)
    if not isinstance(sampler, dict):
        return
    inputs = sampler.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        return
    model_ref = inputs.get("model")
    if not isinstance(model_ref, list) or not model_ref:
        return

    available_nodes = set(available_nodes or ())
    engine = _inpaint_engine(request)
    if engine == "lanpaint":
        has_flux_kontext_reference_method = any(
            str(node.get("class_type", "")).lower() in {"fluxkontextmultireferencelatentmethod", "referencelatent"}
            for node in api.values()
            if isinstance(node, dict)
        )
        if request.preset.lower() == "qwen":
            engine = "default"
        else:
            sampler_class = str(sampler.get("class_type", "") or "")
            if sampler_class.lower() != "ksampler":
                engine = "differential"
            elif available_nodes and "LanPaint_KSampler" not in available_nodes:
                engine = "differential"
            else:
                sampler["class_type"] = "LanPaint_KSampler"
                sampler.setdefault("_meta", {})["title"] = "LanPaint KSampler"
                inputs["LanPaint_NumSteps"] = max(0, min(100, int(getattr(request.img2img, "lanpaint_thinking_steps", 5) or 5)))
                prompt_mode = str(getattr(request.img2img, "lanpaint_prompt_mode", "Image First") or "Image First")
                inputs["LanPaint_PromptMode"] = "Prompt First" if prompt_mode.lower().startswith("prompt") else "Image First"
                inputs["LanPaint_Info"] = "LanPaint KSampler"
                inputs["Inpainting_mode"] = "🖼️ Image Inpainting"
                return

    if engine != "differential":
        return
    if available_nodes and "DifferentialDiffusion" not in available_nodes:
        return
    if any(str(node.get("class_type", "")).lower() == "differentialdiffusion" for node in api.values() if isinstance(node, dict)):
        return
    node_id = str(_next_api_node_id(api))
    api[node_id] = {
        "class_type": "DifferentialDiffusion",
        "inputs": {
            "model": model_ref,
            "strength": max(0.0, min(1.0, float(getattr(request.img2img, "differential_strength", 1.0) or 1.0))),
        },
        "_meta": {"title": "Differential Diffusion Inpaint"},
    }
    inputs["model"] = [node_id, 0]


def _image_scale_node(image_ref: list[Any], width: int | float, height: int | float, method: str = "lanczos") -> dict[str, Any]:
    return {
        "class_type": "ImageScale",
        "inputs": {
            "image": image_ref,
            "width": max(16, int(width)),
            "height": max(16, int(height)),
            "upscale_method": method,
            "crop": "disabled",
        },
        "_meta": {"title": "Resize Reference To Side Menu"},
    }


def _qwen_flux_image_scale_node(image_ref: list[Any], title: str = "QWEN FluxKontext Image Scale") -> dict[str, Any]:
    return {
        "class_type": "FluxKontextImageScale",
        "inputs": {"image": image_ref},
        "_meta": {"title": title},
    }


def _truthy_option(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "off", "none", "no"}
    return bool(value)


def _qwen_multiangle_prompt(horizontal_angle: Any, vertical_angle: Any, zoom: Any) -> str:
    try:
        horizontal = int(round(float(horizontal_angle)))
    except (TypeError, ValueError):
        horizontal = 54
    try:
        vertical = int(round(float(vertical_angle)))
    except (TypeError, ValueError):
        vertical = 29
    try:
        zoom_value = float(zoom)
    except (TypeError, ValueError):
        zoom_value = 2.1
    horizontal %= 360
    vertical = max(-30, min(60, vertical))
    zoom_value = max(0.0, min(10.0, zoom_value))

    if horizontal < 22.5 or horizontal >= 337.5:
        h_direction = "front view"
    elif horizontal < 67.5:
        h_direction = "front-right quarter view"
    elif horizontal < 112.5:
        h_direction = "right side view"
    elif horizontal < 157.5:
        h_direction = "back-right quarter view"
    elif horizontal < 202.5:
        h_direction = "back view"
    elif horizontal < 247.5:
        h_direction = "back-left quarter view"
    elif horizontal < 292.5:
        h_direction = "left side view"
    else:
        h_direction = "front-left quarter view"

    if vertical < -15:
        v_direction = "low-angle shot"
    elif vertical < 15:
        v_direction = "eye-level shot"
    elif vertical < 45:
        v_direction = "elevated shot"
    else:
        v_direction = "high-angle shot"

    if zoom_value < 2:
        distance = "wide shot"
    elif zoom_value < 6:
        distance = "medium shot"
    else:
        distance = "close-up"
    return f"<sks> {h_direction} {v_direction} {distance}"


def _qwen_multiview_options(request: GenerateRequest) -> dict[str, Any]:
    video_options = request.video or {}
    enabled = request.preset.lower() == "qwen" and request.activity == "img2img" and _truthy_option(video_options.get("qwen_multiview"))
    try:
        horizontal = int(round(float(video_options.get("qwen_camera_horizontal", 54)))) % 360
    except (TypeError, ValueError):
        horizontal = 54
    try:
        vertical = max(-30, min(60, int(round(float(video_options.get("qwen_camera_vertical", 29))))))
    except (TypeError, ValueError):
        vertical = 29
    try:
        zoom = max(0.0, min(10.0, float(video_options.get("qwen_camera_zoom", 2.1))))
    except (TypeError, ValueError):
        zoom = 2.1
    return {
        "enabled": enabled,
        "horizontal": horizontal,
        "vertical": vertical,
        "zoom": zoom,
        "camera_view": _truthy_option(video_options.get("qwen_camera_view")),
        "prompt": _qwen_multiangle_prompt(horizontal, vertical, zoom),
    }


def _flux_multiview_options(request: GenerateRequest) -> dict[str, Any]:
    video_options = request.video or {}
    enabled = request.preset.lower() == "flux" and request.activity == "img2img" and _truthy_option(video_options.get("flux_multiview"))
    try:
        horizontal = int(round(float(video_options.get("flux_camera_horizontal", video_options.get("qwen_camera_horizontal", 54))))) % 360
    except (TypeError, ValueError):
        horizontal = 54
    try:
        vertical = max(-30, min(60, int(round(float(video_options.get("flux_camera_vertical", video_options.get("qwen_camera_vertical", 29)))))))
    except (TypeError, ValueError):
        vertical = 29
    try:
        zoom = max(0.0, min(10.0, float(video_options.get("flux_camera_zoom", video_options.get("qwen_camera_zoom", 2.1)))))
    except (TypeError, ValueError):
        zoom = 2.1
    return {
        "enabled": enabled,
        "horizontal": horizontal,
        "vertical": vertical,
        "zoom": zoom,
        "camera_view": _truthy_option(video_options.get("flux_camera_view", video_options.get("qwen_camera_view"))),
        "prompt": _qwen_multiangle_prompt(horizontal, vertical, zoom),
    }


def _qwen_pose_studio_handoff(request: GenerateRequest) -> bool:
    video_options = request.video or {}
    return (
        request.preset.lower() == "qwen"
        and request.activity == "img2img"
        and _truthy_option(video_options.get("pose_studio"))
    )


def _prepend_qwen_camera_prompt(prompt: str, camera_prompt: str) -> str:
    base = str(prompt or "").strip()
    if not camera_prompt:
        return base
    if camera_prompt.lower() in base.lower():
        return base
    return f"{camera_prompt}, {base}" if base else camera_prompt


def _controlnet_can_apply(request: GenerateRequest, controlnet_name: str | None, controlnet_image_name: str | None) -> bool:
    preset = str(request.preset or "").lower()
    return bool(
        request.controlnet.enabled
        and controlnet_name
        and controlnet_image_name
        and preset in {"sd", "sd15", "xl", "sdxl", "flux", "qwen", "anima", "zimageturbo", "zimage"}
    )


def _append_controlnet_preprocessor(
    workflow: dict[str, Any],
    request: GenerateRequest,
    image_ref: list[Any],
    *,
    start_id: int,
    title_prefix: str = "ControlNet",
) -> tuple[list[Any], int]:
    control_type = str(request.controlnet.type or "").lower()
    preprocessor = str(request.controlnet.preprocessor or "auto").lower()
    if _truthy_option((request.video or {}).get("pose_studio")) and control_type in {"dwpose", "openpose", "pose"}:
        return image_ref, start_id
    if preprocessor in {"none", "off", "disabled"}:
        return image_ref, start_id
    if control_type == "canny":
        canny_id = str(start_id)
        workflow[canny_id] = {
            "class_type": "Canny",
            "inputs": {
                "image": image_ref,
                "low_threshold": max(0.01, min(0.99, float(request.controlnet.low_threshold or 0.4))),
                "high_threshold": max(0.01, min(0.99, float(request.controlnet.high_threshold or 0.8))),
            },
            "_meta": {"title": f"{title_prefix} Canny Preprocessor"},
        }
        return [canny_id, 0], start_id + 1
    if control_type in {"dwpose", "openpose", "pose"}:
        pose_id = str(start_id)
        if control_type == "dwpose":
            workflow[pose_id] = {
                "class_type": "DWPreprocessor",
                "inputs": {
                    "image": image_ref,
                    "detect_hand": "enable",
                    "detect_body": "enable",
                    "detect_face": "enable",
                    "resolution": 512,
                    "bbox_detector": "yolox_l.onnx",
                    "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
                    "scale_stick_for_xinsr_cn": "disable",
                },
                "_meta": {"title": f"{title_prefix} DWPose Preprocessor"},
            }
        else:
            workflow[pose_id] = {
                "class_type": "OpenposePreprocessor",
                "inputs": {
                    "image": image_ref,
                    "detect_hand": "enable",
                    "detect_body": "enable",
                    "detect_face": "enable",
                    "resolution": 512,
                },
                "_meta": {"title": f"{title_prefix} OpenPose Preprocessor"},
            }
        return [pose_id, 0], start_id + 1
    if control_type == "depth":
        depth_id = str(start_id)
        workflow[depth_id] = {
            "class_type": "DepthAnythingV2Preprocessor",
            "inputs": {
                "image": image_ref,
                "ckpt_name": "depth_anything_v2_vitl.pth",
                "resolution": 512,
            },
            "_meta": {"title": f"{title_prefix} Depth Preprocessor"},
        }
        return [depth_id, 0], start_id + 1
    if control_type == "lineart":
        lineart_id = str(start_id)
        workflow[lineart_id] = {
            "class_type": "LineArtPreprocessor",
            "inputs": {
                "image": image_ref,
                "coarse": "disable",
                "resolution": 512,
            },
            "_meta": {"title": f"{title_prefix} LineArt Preprocessor"},
        }
        return [lineart_id, 0], start_id + 1
    return image_ref, start_id


def _append_anima_lllite_controlnet_route(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    model_ref: list[Any],
    controlnet_name: str | None,
    controlnet_image_name: str | None,
    start_id: int = 120,
) -> tuple[list[Any], int]:
    if not _controlnet_can_apply(request, controlnet_name, controlnet_image_name):
        return model_ref, start_id
    load_image_id = str(start_id)
    image_ref: list[Any] = [load_image_id, 0]
    next_id = start_id + 1
    workflow[load_image_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": controlnet_image_name},
        "_meta": {"title": "Anima Control Image"},
    }
    control_type = str(request.controlnet.type or "").lower()
    image_ref, next_id = _append_controlnet_preprocessor(workflow, request, image_ref, start_id=next_id, title_prefix="Anima")

    apply_id = str(next_id)
    workflow[apply_id] = {
        "class_type": "AnimaLLLiteApply",
        "inputs": {
            "model": model_ref,
            "lllite_name": controlnet_name,
            "image": image_ref,
            "strength": max(0.0, min(10.0, float(request.controlnet.strength or 1.0))),
            "start_percent": max(0.0, min(1.0, float(request.controlnet.start_percent or 0.0))),
            "end_percent": max(0.0, min(1.0, float(request.controlnet.end_percent or 1.0))),
            "preserve_wrapper": True,
        },
        "_meta": {"title": f"Apply Anima LLLite {control_type or 'image'}"},
    }
    return [apply_id, 0], next_id + 1


def _append_controlnet_route(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    positive_ref: list[Any],
    negative_ref: list[Any],
    controlnet_name: str | None,
    controlnet_image_name: str | None,
    vae_ref: list[Any] | None = None,
    start_id: int = 40,
) -> tuple[list[Any], list[Any], int]:
    if not _controlnet_can_apply(request, controlnet_name, controlnet_image_name):
        return positive_ref, negative_ref, start_id
    load_image_id = str(start_id)
    image_ref: list[Any] = [load_image_id, 0]
    next_id = start_id + 1
    workflow[load_image_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": controlnet_image_name},
        "_meta": {"title": "ControlNet Image"},
    }
    control_type = str(request.controlnet.type or "").lower()
    image_ref, next_id = _append_controlnet_preprocessor(workflow, request, image_ref, start_id=next_id)

    loader_id = str(next_id)
    apply_id = str(next_id + 1)
    workflow[loader_id] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": controlnet_name},
        "_meta": {"title": "Load ControlNet"},
    }
    apply_inputs: dict[str, Any] = {
        "positive": positive_ref,
        "negative": negative_ref,
        "control_net": [loader_id, 0],
        "image": image_ref,
        "strength": max(0.0, min(10.0, float(request.controlnet.strength or 0.75))),
        "start_percent": max(0.0, min(1.0, float(request.controlnet.start_percent or 0.0))),
        "end_percent": max(0.0, min(1.0, float(request.controlnet.end_percent or 1.0))),
    }
    if vae_ref:
        apply_inputs["vae"] = vae_ref
    workflow[apply_id] = {
        "class_type": "ControlNetApplyAdvanced",
        "inputs": apply_inputs,
        "_meta": {"title": f"Apply ControlNet {control_type or 'image'}"},
    }
    return [apply_id, 0], [apply_id, 1], next_id + 2


def _append_zimage_fun_controlnet_route(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    model_ref: list[Any],
    controlnet_name: str | None,
    controlnet_image_name: str | None,
    vae_ref: list[Any],
    start_id: int = 120,
) -> tuple[list[Any], int]:
    if not _controlnet_can_apply(request, controlnet_name, controlnet_image_name):
        return model_ref, start_id
    load_image_id = str(start_id)
    image_ref: list[Any] = [load_image_id, 0]
    next_id = start_id + 1
    workflow[load_image_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": controlnet_image_name},
        "_meta": {"title": "Z-Image Control Image"},
    }
    control_type = str(request.controlnet.type or "").lower()
    image_ref, next_id = _append_controlnet_preprocessor(workflow, request, image_ref, start_id=next_id, title_prefix="Z-Image")

    patch_loader_id = str(next_id)
    apply_id = str(next_id + 1)
    workflow[patch_loader_id] = {
        "class_type": "ModelPatchLoader",
        "inputs": {"name": controlnet_name},
        "_meta": {"title": "Load Z-Image ControlNet Patch"},
    }
    workflow[apply_id] = {
        "class_type": "ZImageFunControlnet",
        "inputs": {
            "model": model_ref,
            "model_patch": [patch_loader_id, 0],
            "vae": vae_ref,
            "strength": max(0.0, min(10.0, float(request.controlnet.strength or 1.0))),
            "image": image_ref,
        },
        "_meta": {"title": f"Apply Z-Image ControlNet {control_type or 'image'}"},
    }
    return [apply_id, 0], next_id + 2


def _append_qwen_model_patch_controlnet_route(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    model_ref: list[Any],
    controlnet_name: str | None,
    controlnet_image_name: str | None,
    vae_ref: list[Any],
    start_id: int = 120,
) -> tuple[list[Any], int]:
    if not _controlnet_can_apply(request, controlnet_name, controlnet_image_name):
        return model_ref, start_id
    load_image_id = str(start_id)
    image_ref: list[Any] = [load_image_id, 0]
    next_id = start_id + 1
    workflow[load_image_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": controlnet_image_name},
        "_meta": {"title": "Qwen Control Image"},
    }
    control_type = str(request.controlnet.type or "").lower()
    image_ref, next_id = _append_controlnet_preprocessor(workflow, request, image_ref, start_id=next_id, title_prefix="Qwen")

    patch_loader_id = str(next_id)
    apply_id = str(next_id + 1)
    workflow[patch_loader_id] = {
        "class_type": "ModelPatchLoader",
        "inputs": {"name": controlnet_name},
        "_meta": {"title": "Load Qwen ControlNet Patch"},
    }
    workflow[apply_id] = {
        "class_type": "QwenImageDiffsynthControlnet",
        "inputs": {
            "model": model_ref,
            "model_patch": [patch_loader_id, 0],
            "vae": vae_ref,
            "image": image_ref,
            "strength": max(0.0, min(10.0, float(request.controlnet.strength or 1.0))),
        },
        "_meta": {"title": f"Apply Qwen ControlNet {control_type or 'image'}"},
    }
    return [apply_id, 0], next_id + 2


def _append_flux_union_controlnet_route(
    workflow: dict[str, Any],
    request: GenerateRequest,
    *,
    positive_ref: list[Any],
    negative_ref: list[Any],
    controlnet_name: str | None,
    controlnet_image_name: str | None,
    vae_ref: list[Any],
    start_id: int = 120,
) -> tuple[list[Any], list[Any], int]:
    if not _controlnet_can_apply(request, controlnet_name, controlnet_image_name):
        return positive_ref, negative_ref, start_id
    load_image_id = str(start_id)
    image_ref: list[Any] = [load_image_id, 0]
    next_id = start_id + 1
    workflow[load_image_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": controlnet_image_name},
        "_meta": {"title": "Flux Control Image"},
    }
    control_type = str(request.controlnet.type or "canny").lower()
    image_ref, next_id = _append_controlnet_preprocessor(workflow, request, image_ref, start_id=next_id, title_prefix="Flux")
    union_type = {
        "openpose": "pose",
        "dwpose": "pose",
        "pose": "pose",
        "depth": "depth",
        "tile": "tile",
        "blur": "blur",
        "gray": "gray",
        "greyscale": "gray",
        "low quality": "low quality",
        "low_quality": "low quality",
    }.get(control_type, "canny")

    loader_id = str(next_id)
    apply_id = str(next_id + 1)
    workflow[loader_id] = {
        "class_type": "ControlNetLoader",
        "inputs": {"control_net_name": controlnet_name},
        "_meta": {"title": "Load Flux ControlNet"},
    }
    workflow[apply_id] = {
        "class_type": "FluxUnionControlNetApply",
        "inputs": {
            "conditioning": positive_ref,
            "control_net": [loader_id, 0],
            "image": image_ref,
            "union_controlnet_type": union_type,
            "strength": max(0.0, min(10.0, float(request.controlnet.strength or 0.75))),
            "start_percent": max(0.0, min(1.0, float(request.controlnet.start_percent or 0.0))),
            "end_percent": max(0.0, min(1.0, float(request.controlnet.end_percent or 1.0))),
            "vae": vae_ref,
        },
        "_meta": {"title": f"Apply Flux ControlNet {union_type}"},
    }
    return [apply_id, 0], negative_ref, next_id + 2


def build_basic_sd_workflow(
    request: GenerateRequest,
    checkpoint_name: str,
    reference_image_name: str | None = None,
    mask_image_name: str | None = None,
    controlnet_name: str | None = None,
    controlnet_image_name: str | None = None,
    vae_name: str | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    latent_node = ["4", 0]
    denoise = request.denoise
    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["1", 1]
    vae_ref: list[Any] = ["30", 0] if vae_name else ["1", 2]
    workflow = {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.prompt, "clip": clip_ref},
        },
        "3": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.negative_prompt, "clip": clip_ref},
        },
        "4": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": request.width,
                "height": request.height,
                "batch_size": max(1, request.batch_size),
            },
        },
        "5": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": normalize_sampler(request.sampler),
                "scheduler": normalize_scheduler(request.scheduler),
                "denoise": denoise,
                "model": model_ref,
                "positive": ["2", 0],
                "negative": ["3", 0],
                "latent_image": latent_node,
            },
        },
        "6": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["5", 0], "vae": vae_ref},
        },
        "7": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "NEXUS_BTA", "images": ["6", 0]},
        },
    }
    if vae_name:
        workflow["30"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Side Menu VAE"},
        }
    model_ref, clip_ref, _ = _append_lora_chain(workflow, request, ["1", 0], ["1", 1], start_id=20)
    workflow["2"]["inputs"]["clip"] = clip_ref
    workflow["3"]["inputs"]["clip"] = clip_ref
    workflow["5"]["inputs"]["model"] = model_ref
    positive_ref, negative_ref, _ = _append_controlnet_route(
        workflow,
        request,
        positive_ref=["2", 0],
        negative_ref=["3", 0],
        controlnet_name=controlnet_name,
        controlnet_image_name=controlnet_image_name,
        vae_ref=vae_ref,
        start_id=40,
    )
    workflow["5"]["inputs"]["positive"] = positive_ref
    workflow["5"]["inputs"]["negative"] = negative_ref
    if reference_image_name:
        denoise = request.img2img.denoise
        workflow["4"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
        }
        workflow["9"] = _image_scale_node(["4", 0], request.width, request.height)
        workflow["8"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["9", 0], "vae": vae_ref},
        }
        workflow["5"]["inputs"]["latent_image"] = ["8", 0]
        workflow["5"]["inputs"]["denoise"] = denoise
        _append_inpaint_mask(
            workflow,
            request,
            reference_node_id="4",
            vae_ref=vae_ref,
            sampler_node_id="5",
            mask_image_name=mask_image_name,
            decoded_image_ref=["6", 0],
            save_node_id="7",
        )
        ensure_inpaint_engine_route(workflow, request)
    return workflow


def build_basic_anima_workflow(
    request: GenerateRequest,
    model_name: str,
    text_encoder_name: str | None = None,
    vae_name: str | None = None,
    reference_image_name: str | None = None,
    mask_image_name: str | None = None,
    controlnet_name: str | None = None,
    controlnet_image_name: str | None = None,
) -> dict[str, Any]:
    if not text_encoder_name:
        raise ValueError("Anima requires anima2BQwen354BText_base.safetensors or another Qwen-compatible text encoder in models/text_encoders.")
    if not vae_name:
        raise ValueError("Anima requires qwen_image_vae.safetensors in models/vae.")

    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    model_loader_class = "UnetLoaderGGUF" if str(model_name or "").lower().endswith(".gguf") else "UNETLoader"
    model_loader_inputs: dict[str, Any] = {"unet_name": model_name}
    if model_loader_class == "UNETLoader":
        model_loader_inputs["weight_dtype"] = "default"

    model_ref: list[Any] = ["1", 0]
    clip_ref: list[Any] = ["2", 0]
    vae_ref: list[Any] = ["3", 0]
    latent_ref: list[Any] = ["6", 0]
    denoise = request.denoise
    workflow: dict[str, Any] = {
        "1": {
            "class_type": model_loader_class,
            "inputs": model_loader_inputs,
            "_meta": {"title": "Load Anima Diffusion Model"},
        },
        "2": _anima_clip_loader_node(text_encoder_name, "Anima Qwen3 Text Encoder"),
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Anima Qwen Image VAE"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.prompt, "clip": clip_ref},
            "_meta": {"title": "Anima Positive Prompt"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.negative_prompt, "clip": clip_ref},
            "_meta": {"title": "Anima Negative Prompt"},
        },
        "6": {
            "class_type": "EmptyLatentImage",
            "inputs": {
                "width": request.width,
                "height": request.height,
                "batch_size": max(1, request.batch_size),
            },
            "_meta": {"title": "Anima Latent"},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": normalize_sampler(request.sampler),
                "scheduler": normalize_scheduler(request.scheduler),
                "denoise": denoise,
                "model": model_ref,
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": latent_ref,
            },
            "_meta": {"title": "Anima Sampler"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": vae_ref},
            "_meta": {"title": "Anima VAE Decode"},
        },
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "NEXUS_BTA_ANIMA", "images": ["8", 0]},
            "_meta": {"title": "Save Anima Image"},
        },
    }

    model_ref, clip_ref, _ = _append_lora_chain(workflow, request, model_ref, clip_ref, start_id=20)
    model_ref, _ = _append_anima_lllite_controlnet_route(
        workflow,
        request,
        model_ref=model_ref,
        controlnet_name=controlnet_name,
        controlnet_image_name=controlnet_image_name,
    )
    workflow["4"]["inputs"]["clip"] = clip_ref
    workflow["5"]["inputs"]["clip"] = clip_ref
    workflow["7"]["inputs"]["model"] = model_ref

    if reference_image_name:
        workflow["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
            "_meta": {"title": "Anima Reference Image"},
        }
        workflow["11"] = _image_scale_node(["10", 0], request.width, request.height)
        workflow["12"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["11", 0], "vae": vae_ref},
            "_meta": {"title": "Encode Anima Reference"},
        }
        workflow["7"]["inputs"]["latent_image"] = ["12", 0]
        workflow["7"]["inputs"]["denoise"] = request.img2img.denoise
        _append_inpaint_mask(
            workflow,
            request,
            reference_node_id="10",
            vae_ref=vae_ref,
            sampler_node_id="7",
            mask_image_name=mask_image_name,
            decoded_image_ref=["8", 0],
            save_node_id="9",
        )
        ensure_inpaint_engine_route(workflow, request)

    return workflow


def build_basic_qwen_image_workflow(
    request: GenerateRequest,
    checkpoint_name: str,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str | None = None,
    reference_image_names: list[str] | None = None,
    mask_image_name: str | None = None,
    controlnet_name: str | None = None,
    controlnet_image_name: str | None = None,
    controlnet_category: str | None = None,
    available_nodes: set[str] | None = None,
) -> dict[str, Any]:
    _apply_outpaint_continuity_prompt(request)
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    width = max(16, int(request.width))
    height = max(16, int(request.height))
    available_nodes = set(available_nodes or ())
    refs = [name for name in (reference_image_names or ([reference_image_name] if reference_image_name else [])) if name][:3]
    reference_image_name = refs[0] if refs else None
    is_qwen_inpaint = bool(reference_image_name and mask_image_name and _uses_inpaint_mask_mode(request))
    qwen_batch_size = 1 if reference_image_name else max(1, request.batch_size)
    qwen_denoise = 1.0 if _uses_outpaint_extend_mode(request) else request.img2img.denoise
    qwen_multiview = _qwen_multiview_options(request)
    qwen_pose_studio = _qwen_pose_studio_handoff(request)
    qwen_pose_controlnet = bool(qwen_pose_studio and controlnet_name and controlnet_image_name)
    if qwen_pose_studio and refs and qwen_pose_controlnet:
        refs = refs[:1]
        reference_image_name = refs[0]
    elif qwen_pose_studio and refs:
        refs = refs[:2]
        reference_image_name = refs[0]
    if qwen_multiview["enabled"]:
        qwen_denoise = 1.0
    if qwen_pose_studio:
        qwen_denoise = 1.0
    qwen_linear_view = (
        bool(refs)
        and _truthy_option((request.video or {}).get("qwen_linear_view"))
        and not qwen_multiview["enabled"]
        and not qwen_pose_studio
        and not is_qwen_inpaint
    )
    qwen_reference_method_available = "FluxKontextMultiReferenceLatentMethod" in available_nodes
    loader_class = "UnetLoaderGGUF" if checkpoint_name.lower().endswith(".gguf") else "UNETLoader"
    loader_inputs = (
        {"unet_name": checkpoint_name}
        if loader_class == "UnetLoaderGGUF"
        else {"unet_name": checkpoint_name, "weight_dtype": "default"}
    )
    model_ref: list[Any] = ["1", 0]
    qwen_lora_nodes: dict[str, Any] = {}
    next_lora_id = 20
    for lora_name, strength_model, _strength_clip in _active_lora_selections(request):
        node_id = str(next_lora_id)
        qwen_lora_nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora_name,
                "strength_model": float(strength_model),
            },
            "_meta": {"title": f"QWEN LoRA - {Path(lora_name).name}"},
        }
        model_ref = [node_id, 0]
        next_lora_id += 1

    prompt_text = str(qwen_multiview["prompt"]) if qwen_multiview["enabled"] else request.prompt
    negative_prompt_text = "" if qwen_multiview["enabled"] else (request.negative_prompt or "")
    if qwen_pose_studio:
        if qwen_pose_controlnet:
            prompt_prefix = "Use the ControlNet POSE/OpenPose/DWPose guide as the only body pose and composition target"
            if refs:
                prompt_prefix += ", and use Image 1 only for character identity/appearance"
            prompt_prefix += ". Render the full body from head to feet inside the requested frame, do not crop the character, do not copy the original reference image pose, do not render the colored skeleton/guide marks, and match the ControlNet pose guide composition exactly"
        else:
            prompt_prefix = "Use Image 1 as the POSE Studio OpenPose/DWPose body pose and composition guide"
            if len(refs) >= 2:
                prompt_prefix += ", and use Image 2 only for character identity and appearance"
            prompt_prefix += ". Render the full body from head to feet inside the requested frame, do not crop the character, do not copy the original reference image pose, do not render the colored skeleton/guide marks, and match Image 1 pose guide composition exactly"
        prompt_text = prompt_prefix + ". " + str(prompt_text or request.prompt or "")
        negative_additions = (
            "cropped body, close-up crop, torso crop, cut off head, cut off feet, missing legs, "
            "missing arms, ignored pose guide, copied reference pose, stretched character, rendered skeleton marks"
        )
        negative_prompt_text = (
            f"{negative_additions}, {negative_prompt_text}"
            if negative_prompt_text.strip()
            else negative_additions
        )
    elif len(refs) > 1 and not qwen_multiview["enabled"]:
        prompt_prefix = "Use Image 1 as the base reference"
        if len(refs) >= 2:
            prompt_prefix += ", Image 2 as the secondary reference"
        if len(refs) >= 3:
            prompt_prefix += ", and Image 3 as the additional reference"
        prompt_text = prompt_prefix + ". " + str(prompt_text or request.prompt or "")

    workflow = {
        "1": {
            "class_type": loader_class,
            "inputs": loader_inputs,
            "_meta": {"title": "Load QWEN Model"},
        },
        "2": _clip_loader_node(text_encoder_name, "qwen_image", "QWEN Text Encoder"),
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "QWEN VAE"},
        },
        "8": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": model_ref, "shift": float((request.video or {}).get("shift") or 3.1)},
            "_meta": {"title": "QWEN AuraFlow Sampling"},
        },
        "9": {
            "class_type": "CFGNorm",
            "inputs": {"model": ["8", 0], "strength": 1.0},
            "_meta": {"title": "QWEN CFG Norm"},
        },
        "10": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": request.steps,
                "cfg": request.cfg,
                "sampler_name": normalize_sampler(request.sampler),
                "scheduler": normalize_scheduler(request.scheduler),
                "denoise": qwen_denoise if reference_image_name else request.denoise,
                "model": ["9", 0],
                "positive": ["5", 0],
                "negative": ["6", 0],
                "latent_image": ["7", 0],
            },
            "_meta": {"title": "KSampler"},
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["10", 0], "vae": ["3", 0]},
            "_meta": {"title": "VAE Decode"},
        },
        "13": _image_scale_node(["11", 0], int(request.width), int(request.height)),
        "12": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "NEXUS_BTA_QWEN", "images": ["13", 0]},
            "_meta": {"title": "Save Image"},
        },
    }
    workflow["13"]["_meta"]["title"] = "Resize QWEN Output To Frontend Size"
    if is_qwen_inpaint:
        workflow["10"]["class_type"] = "LanPaint_KSampler"
        workflow["10"]["inputs"].update(
            {
                "LanPaint_NumSteps": max(1, int(getattr(request.img2img, "lanpaint_thinking_steps", 1) or 1)),
                "LanPaint_PromptMode": str(getattr(request.img2img, "lanpaint_prompt_mode", "") or "Image First"),
                "LanPaint_Info": "",
                "Inpainting_mode": "🖼️ Image Inpainting",
            }
        )
        workflow["10"]["_meta"]["title"] = "QWEN LanPaint KSampler"
    workflow.update(qwen_lora_nodes)
    qwen_base_image_ref: list[Any] | None = None
    qwen_conditioning_image_refs: dict[int, list[Any]] = {}
    qwen_extend_mask_ref: list[Any] | None = None

    if reference_image_name:
        qwen_extend_pad = is_qwen_inpaint and _uses_outpaint_extend_mode(request) and _has_extend_pad(request)
        for index, name in enumerate(refs, start=1):
            load_id = "4" if index == 1 else str(39 + index)
            scale_id = str(59 + index)
            workflow[load_id] = {
                "class_type": "LoadImage",
                "inputs": {"image": name},
                "_meta": {"title": f"Reference Image {index}"},
            }
            if qwen_extend_pad and index == 1:
                if _uses_prepadded_outpaint_reference(request):
                    workflow[scale_id] = _image_scale_node([load_id, 0], width, height)
                    workflow[scale_id]["_meta"]["title"] = "QWEN Pre-Padded Extend Reference"
                    qwen_conditioning_image_refs[index] = [scale_id, 0]
                else:
                    workflow[scale_id] = _image_pad_for_outpaint_node([load_id, 0], request, feathering=40, available_nodes=available_nodes)
                    workflow[scale_id]["_meta"]["title"] = "QWEN Pad Base For Extend"
                    qwen_conditioning_image_refs[index] = [scale_id, 0]
                    qwen_extend_mask_ref = [scale_id, 1]
            elif qwen_pose_studio:
                workflow[scale_id] = _image_scale_node([load_id, 0], width, height)
                workflow[scale_id]["_meta"]["title"] = f"Resize QWEN POSE Reference {index} To Output Frame"
            elif qwen_linear_view:
                workflow[scale_id] = _qwen_flux_image_scale_node([load_id, 0], f"QWEN Linear Reference {index} FluxKontext Scale")
            elif not is_qwen_inpaint:
                workflow[scale_id] = _image_scale_node([load_id, 0], width, height)
                workflow[scale_id]["_meta"]["title"] = (
                    "Resize QWEN Linear Reference To Side Menu"
                    if qwen_linear_view and index == 1
                    else f"Resize QWEN Reference {index} To Side Menu"
                )
            else:
                workflow[scale_id] = _image_scale_node([load_id, 0], width, height)
                workflow[scale_id]["_meta"]["title"] = f"Resize QWEN Reference {index} To Side Menu"
        if qwen_multiview["enabled"]:
            workflow["110"] = {
                "class_type": "QwenMultiangleCameraNode",
                "inputs": {
                    "horizontal_angle": int(round(float(qwen_multiview["horizontal"]))),
                    "vertical_angle": int(round(float(qwen_multiview["vertical"]))),
                    "zoom": float(qwen_multiview["zoom"]),
                    "default_prompts": False,
                    "camera_view": bool(qwen_multiview["camera_view"]),
                    "image": ["60", 0],
                },
                "_meta": {"title": "Qwen Multiangle Camera"},
            }
        qwen_prompt_input: Any = ["110", 0] if qwen_multiview["enabled"] else prompt_text
        if len(refs) == 1:
            qwen_base_image_ref = ["60", 0]
            qwen_text_image_ref = qwen_conditioning_image_refs.get(1, qwen_base_image_ref)
            if qwen_pose_studio:
                workflow["7"] = {
                    "class_type": "EmptySD3LatentImage",
                    "inputs": {"width": width, "height": height, "batch_size": qwen_batch_size},
                    "_meta": {"title": "QWEN POSE Studio Empty Latent"},
                }
            elif qwen_linear_view:
                workflow["7"] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": qwen_base_image_ref, "vae": ["3", 0]},
                    "_meta": {"title": "QWEN Linear Encode Base Reference"},
                }
            else:
                workflow["7"] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": qwen_base_image_ref, "vae": ["3", 0]},
                    "_meta": {"title": "Encode QWEN Base Reference"},
                }
            workflow["10"]["inputs"]["latent_image"] = ["7", 0]
            qwen_text_encoder_class = "TextEncodeQwenImageEditPlus" if qwen_linear_view else ("TextEncodeQwenImageEdit" if (qwen_pose_studio or len(refs) == 1) else "TextEncodeQwenImageEditPlus")
            if qwen_linear_view:
                qwen_positive_inputs = {"clip": ["2", 0], "prompt": qwen_prompt_input, "vae": ["3", 0], "image1": qwen_text_image_ref}
                qwen_negative_inputs = {"clip": ["2", 0], "prompt": negative_prompt_text, "vae": ["3", 0], "image1": qwen_text_image_ref}
            else:
                qwen_positive_inputs = (
                    {"clip": ["2", 0], "prompt": qwen_prompt_input, "vae": ["3", 0], "image": qwen_text_image_ref}
                    if (qwen_pose_studio or len(refs) == 1)
                    else {"clip": ["2", 0], "prompt": qwen_prompt_input, "vae": ["3", 0], "image1": qwen_text_image_ref}
                )
                qwen_negative_inputs = {"clip": ["2", 0], "prompt": negative_prompt_text, "vae": ["3", 0]}
            workflow["5"] = {
                "class_type": qwen_text_encoder_class,
                "inputs": qwen_positive_inputs,
                "_meta": {"title": "Positive Prompt"},
            }
            workflow["6"] = {
                "class_type": qwen_text_encoder_class,
                "inputs": qwen_negative_inputs,
                "_meta": {"title": "QWEN Negative Prompt"},
            }
            workflow["10"]["inputs"]["positive"] = ["5", 0]
            workflow["10"]["inputs"]["negative"] = ["6", 0]
            if qwen_reference_method_available and qwen_linear_view:
                workflow["15"] = {
                    "class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["5", 0], "reference_latents_method": "index_timestep_zero"},
                    "_meta": {"title": "QWEN Reference Method"},
                }
                workflow["16"] = {
                    "class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"},
                    "_meta": {"title": "QWEN Negative Reference Method"},
                }
                workflow["10"]["inputs"]["positive"] = ["15", 0]
                workflow["10"]["inputs"]["negative"] = ["16", 0]
        else:
            qwen_base_image_ref = ["60", 0]
            if qwen_pose_studio:
                workflow["7"] = {
                    "class_type": "EmptySD3LatentImage",
                    "inputs": {"width": width, "height": height, "batch_size": qwen_batch_size},
                    "_meta": {"title": "QWEN POSE Studio Empty Latent"},
                }
            elif qwen_linear_view:
                workflow["7"] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": qwen_base_image_ref, "vae": ["3", 0]},
                    "_meta": {"title": "QWEN Linear Encode Base Reference"},
                }
            else:
                workflow["7"] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": ["60", 0], "vae": ["3", 0]},
                    "_meta": {"title": "Encode QWEN Base Reference"},
                }
            workflow["10"]["inputs"]["latent_image"] = ["7", 0]
            positive_inputs = {"clip": ["2", 0], "prompt": qwen_prompt_input, "vae": ["3", 0]}
            negative_inputs = {"clip": ["2", 0], "prompt": negative_prompt_text, "vae": ["3", 0]}
            for index, _name in enumerate(refs, start=1):
                image_ref = qwen_conditioning_image_refs.get(index, [str(59 + index), 0])
                positive_inputs[f"image{index}"] = image_ref
                if qwen_linear_view:
                    negative_inputs[f"image{index}"] = image_ref
            workflow["5"] = {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": positive_inputs,
                "_meta": {"title": "Positive Prompt"},
            }
            workflow["6"] = {
                "class_type": "TextEncodeQwenImageEditPlus",
                "inputs": negative_inputs,
                "_meta": {"title": "QWEN Negative Prompt"},
            }
            workflow["10"]["inputs"]["positive"] = ["5", 0]
            workflow["10"]["inputs"]["negative"] = ["6", 0]
            if qwen_reference_method_available and qwen_linear_view:
                workflow["15"] = {
                    "class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["5", 0], "reference_latents_method": "index_timestep_zero"},
                    "_meta": {"title": "QWEN Reference Method"},
                }
                workflow["16"] = {
                    "class_type": "FluxKontextMultiReferenceLatentMethod",
                    "inputs": {"conditioning": ["6", 0], "reference_latents_method": "index_timestep_zero"},
                    "_meta": {"title": "QWEN Negative Reference Method"},
                }
                workflow["10"]["inputs"]["positive"] = ["15", 0]
                workflow["10"]["inputs"]["negative"] = ["16", 0]
    else:
        workflow["5"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.prompt, "clip": ["2", 0]},
            "_meta": {"title": "Positive Prompt"},
        }
        workflow["6"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.negative_prompt, "clip": ["2", 0]},
            "_meta": {"title": "Negative Prompt"},
        }
        workflow["7"] = {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": max(1, request.batch_size)},
            "_meta": {"title": "QWEN Latent"},
        }
    if reference_image_name and mask_image_name and _uses_inpaint_mask_mode(request):
        if is_qwen_inpaint:
            _append_qwen_inpaint_noise_mask(
                workflow,
                request,
                base_latent_ref=["7", 0],
                sampler_node_id="10",
                mask_image_name=mask_image_name,
                mask_ref_override=qwen_extend_mask_ref,
                base_image_ref=qwen_base_image_ref,
                decoded_image_ref=["13", 0],
                save_node_id="12",
                available_nodes=available_nodes,
            )
        else:
            _append_inpaint_mask(
                workflow,
                request,
                reference_node_id="4",
                vae_ref=["3", 0],
                sampler_node_id="10",
                mask_image_name=mask_image_name,
                decoded_image_ref=["13", 0],
                save_node_id="12",
            )
    if controlnet_category == "model_patches" and not qwen_pose_studio:
        patched_model_ref, _ = _append_qwen_model_patch_controlnet_route(
            workflow,
            request,
            model_ref=list(workflow["8"]["inputs"]["model"]),
            controlnet_name=controlnet_name,
            controlnet_image_name=controlnet_image_name,
            vae_ref=["3", 0],
            start_id=120,
        )
        workflow["8"]["inputs"]["model"] = patched_model_ref
    elif qwen_pose_studio and not qwen_pose_controlnet:
        pass
    else:
        positive_ref, negative_ref, _ = _append_controlnet_route(
            workflow,
            request,
            positive_ref=list(workflow["10"]["inputs"]["positive"]),
            negative_ref=list(workflow["10"]["inputs"]["negative"]),
            controlnet_name=controlnet_name,
            controlnet_image_name=controlnet_image_name,
            vae_ref=["3", 0],
            start_id=120,
        )
        workflow["10"]["inputs"]["positive"] = positive_ref
        workflow["10"]["inputs"]["negative"] = negative_ref
    return workflow


def build_basic_zimage_turbo_workflow(
    request: GenerateRequest,
    model_name: str,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str | None = None,
    mask_image_name: str | None = None,
    controlnet_name: str | None = None,
    controlnet_image_name: str | None = None,
    controlnet_category: str | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    width = max(16, int(request.width))
    height = max(16, int(request.height))
    width -= width % 16
    height -= height % 16
    denoise = request.img2img.denoise if reference_image_name else request.denoise
    sampler = normalize_sampler(request.sampler or "res_multistep")
    scheduler = normalize_scheduler(request.scheduler or "simple")
    model_ref: list[Any] = ["1", 0]
    lora_nodes: dict[str, Any] = {}
    next_lora_id = 20
    for lora_name, strength_model, _strength_clip in _active_lora_selections(request):
        node_id = str(next_lora_id)
        lora_nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora_name,
                "strength_model": float(strength_model),
            },
            "_meta": {"title": f"Z-Image LoRA - {Path(lora_name).name}"},
        }
        model_ref = [node_id, 0]
        next_lora_id += 1

    positive_inputs: dict[str, Any]
    if reference_image_name:
        positive_inputs = {
            "clip": ["2", 0],
            "prompt": request.prompt,
            "auto_resize_images": True,
            "vae": ["3", 0],
            "image1": ["9", 0],
        }
        positive_class = "TextEncodeZImageOmni"
    else:
        positive_inputs = {"clip": ["2", 0], "text": request.prompt}
        positive_class = "CLIPTextEncode"

    workflow: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader" if not model_name.lower().endswith(".gguf") else "UnetLoaderGGUF",
            "inputs": (
                {"unet_name": model_name, "weight_dtype": "default"}
                if not model_name.lower().endswith(".gguf")
                else {"unet_name": model_name}
            ),
            "_meta": {"title": "Load Z-Image Turbo Model"},
        },
        "2": _clip_loader_node(text_encoder_name, "lumina2", "Z-Image Qwen3 Text Encoder"),
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Z-Image AE VAE"},
        },
        "4": {
            "class_type": positive_class,
            "inputs": positive_inputs,
            "_meta": {"title": "Positive Prompt"},
        },
        "5": (
            {
                "class_type": "CLIPTextEncode",
                "inputs": {"clip": ["2", 0], "text": request.negative_prompt},
                "_meta": {"title": "Negative Prompt"},
            }
            if request.negative_prompt.strip()
            else {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["4", 0]},
                "_meta": {"title": "Official Z-Image Empty Negative"},
            }
        ),
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": max(1, request.batch_size)},
            "_meta": {"title": "Z-Image Latent"},
        },
        "7": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": model_ref, "shift": float((request.video or {}).get("shift") or 3.0)},
            "_meta": {"title": "Z-Image AuraFlow Sampling Shift"},
        },
        "8": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": max(1, int(request.steps or 8)),
                "cfg": request.cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["7", 0],
                "positive": ["4", 0],
                "negative": ["5", 0],
                "latent_image": ["6", 0],
            },
            "_meta": {"title": "Z-Image Turbo Sampler"},
        },
        "11": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["8", 0], "vae": ["3", 0]},
            "_meta": {"title": "VAE Decode"},
        },
        "12": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "NEXUS_BTA_ZIMAGE_TURBO", "images": ["11", 0]},
            "_meta": {"title": "Save Image"},
        },
    }
    workflow.update(lora_nodes)
    if reference_image_name:
        workflow["9"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
            "_meta": {"title": "Reference Image"},
        }
        workflow["13"] = _image_scale_node(["9", 0], width, height)
        workflow["10"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["13", 0], "vae": ["3", 0]},
            "_meta": {"title": "Encode Reference"},
        }
        workflow["8"]["inputs"]["latent_image"] = ["10", 0]
        _append_inpaint_mask(
            workflow,
            request,
            reference_node_id="9",
            vae_ref=["3", 0],
            sampler_node_id="8",
            mask_image_name=mask_image_name,
            decoded_image_ref=["11", 0],
            save_node_id="12",
        )
    if controlnet_category == "model_patches":
        patched_model_ref, _ = _append_zimage_fun_controlnet_route(
            workflow,
            request,
            model_ref=list(workflow["8"]["inputs"]["model"]),
            controlnet_name=controlnet_name,
            controlnet_image_name=controlnet_image_name,
            vae_ref=["3", 0],
            start_id=120,
        )
        workflow["8"]["inputs"]["model"] = patched_model_ref
    else:
        positive_ref, negative_ref, _ = _append_controlnet_route(
            workflow,
            request,
            positive_ref=list(workflow["8"]["inputs"]["positive"]),
            negative_ref=list(workflow["8"]["inputs"]["negative"]),
            controlnet_name=controlnet_name,
            controlnet_image_name=controlnet_image_name,
            vae_ref=["3", 0],
            start_id=120,
        )
        workflow["8"]["inputs"]["positive"] = positive_ref
        workflow["8"]["inputs"]["negative"] = negative_ref
    return workflow


def _ideogram4_has_reference_image(request: GenerateRequest) -> bool:
    return bool(
        (getattr(request.img2img, "reference_image", "") or "").strip()
        or any(bool((value or "").strip()) for value in (getattr(request.img2img, "reference_images", None) or []))
    )


def _ideogram4_effective_side_prompt(request: GenerateRequest) -> str:
    side_prompt = request.prompt.strip()
    if side_prompt:
        return side_prompt
    if request.activity.lower() == "img2img" and _ideogram4_has_reference_image(request):
        return "preserve image"
    return ""


def _ideogram4_text_region_values(request: GenerateRequest) -> list[str]:
    regions = (request.video or {}).get("ideogram_regions")
    if not isinstance(regions, list):
        return []
    values: list[str] = []
    for item in regions[:24]:
        if not isinstance(item, dict):
            continue
        element_type = str(item.get("type") or "obj").strip().lower()
        text = str(item.get("text") or "").strip()
        prompt = str(item.get("prompt") or item.get("desc") or "").strip()
        if element_type == "text" or text:
            value = text or prompt
            if value:
                values.append(value)
    return values


def _ideogram4_requested_brand_marks(request: GenerateRequest) -> bool:
    keywords = (
        "brand",
        "branding",
        "logo",
        "logotype",
        "marca",
        "logomarca",
        "logotipo",
        "watermark",
        "assinatura",
        "signature",
        "emblem",
        "badge",
        "label",
        "etiqueta",
        "sinalizacao",
        "sinalização",
        "signage",
        "sign",
        "placa",
        "poster",
        "cartaz",
    )
    parts = [request.prompt]
    regions = (request.video or {}).get("ideogram_regions")
    if isinstance(regions, list):
        for item in regions[:24]:
            if not isinstance(item, dict):
                continue
            parts.extend(
                [
                    str(item.get("prompt") or ""),
                    str(item.get("desc") or ""),
                    str(item.get("text") or ""),
                    str(item.get("color") or ""),
                ]
            )
    text = " ".join(str(part or "") for part in parts).lower()
    return any(re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text, flags=re.IGNORECASE) for keyword in keywords)


def _ideogram4_clean_output_directive(request: GenerateRequest) -> str:
    text_values = _ideogram4_text_region_values(request)
    allow_brand_marks = _ideogram4_requested_brand_marks(request)
    common = (
        "Create a natural final image that fills the full canvas edge to edge. "
        "Render only the requested subjects, scene details, and explicitly requested text. "
    )
    if not text_values:
        mark_rule = (
            "Include only the requested brand, label, sign, or mark details described by the prompt or regional edits. "
            "Keep unrelated graphic marks and UI-like annotation artifacts out of the scene."
            if allow_brand_marks
            else "Keep the scene free of extra UI-like overlays, annotation artifacts, unrelated graphic marks, and stray writing."
        )
        return f"{common}{mark_rule}"
    mark_rule = (
        "Include only the requested brand, label, sign, or mark details described by the prompt or regional edits. "
        "Keep unrelated graphic marks and UI-like annotation artifacts out of the scene."
        if allow_brand_marks
        else "Keep unrelated writing, extra UI-like overlays, annotation artifacts, and unrelated graphic marks out of the scene."
    )
    quoted = ", ".join(json.dumps(value, ensure_ascii=False) for value in text_values[:8])
    return f"{common}{mark_rule} The only other visible writing must be the exact requested text region words: {quoted}."


def _ideogram4_join_prompt_parts(*parts: str) -> str:
    return " ".join(part.strip() for part in parts if str(part or "").strip())


def _ideogram4_scene_element_desc(side_prompt: str, preserve_image: bool = False) -> str:
    if preserve_image:
        return (
            "Preserve the source image composition, subject identity, lighting, background, and camera framing. "
            "Apply only the explicit regional edits from the other elements."
        )
    prompt = str(side_prompt or "").strip().rstrip(".")
    return f"Overall scene and style direction: {prompt}." if prompt else "Natural coherent scene."


def _ideogram4_regions_prompt(request: GenerateRequest) -> str:
    structured_prompt = parse_ideogram4_prompt_json(request.prompt)
    if structured_prompt is not None:
        return ideogram4_prompt_json_text(structured_prompt)
    regions = (request.video or {}).get("ideogram_regions")
    side_prompt = _ideogram4_effective_side_prompt(request)
    preserve_image = side_prompt.lower() == "preserve image"
    clean_directive = _ideogram4_clean_output_directive(request)
    if (not isinstance(regions, list) or not regions) and not side_prompt:
        return _ideogram4_join_prompt_parts(request.prompt, clean_directive)
    if (not isinstance(regions, list) or not regions) and preserve_image:
        return _ideogram4_join_prompt_parts(side_prompt, clean_directive)

    def _region_unit(value: float | None, default: float = 0.0) -> float:
        if value is None:
            return default
        number = float(value)
        if number > 1.0:
            number /= 100.0
        return max(0.0, min(1.0, number))

    elements: list[dict[str, Any]] = []
    if side_prompt:
        elements.append(
            {
                "type": "obj",
                "bbox": [0, 0, 1000, 1000],
                "desc": _ideogram4_scene_element_desc(side_prompt, preserve_image),
            }
        )
    for item in (regions if isinstance(regions, list) else [])[:24]:
        if not isinstance(item, dict):
            continue
        raw_x = _number_or_none(item.get("x"))
        raw_y = _number_or_none(item.get("y"))
        raw_w = _number_or_none(item.get("w"))
        raw_h = _number_or_none(item.get("h"))
        if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
            continue
        x = _region_unit(raw_x)
        y = _region_unit(raw_y)
        w = max(0.01, min(1.0 - x, _region_unit(raw_w, 0.25)))
        h = max(0.01, min(1.0 - y, _region_unit(raw_h, 0.25)))
        desc = str(item.get("prompt") or item.get("desc") or "").strip()
        text = str(item.get("text") or "").strip()
        element_type = str(item.get("type") or "obj").strip().lower()
        bbox = [
            int(round(y * 1000)),
            int(round(x * 1000)),
            int(round((y + h) * 1000)),
            int(round((x + w) * 1000)),
        ]
        colors = item.get("colors")
        color_palette = [str(color).strip().upper() for color in colors if str(color).strip()] if isinstance(colors, list) else []
        if not color_palette and item.get("color"):
            color_palette = [str(item.get("color")).strip().upper()]
        if element_type == "text" or text:
            display_text = text or desc or "TEXT"
            element = {
                "type": "text",
                "bbox": bbox,
                "text": display_text,
                "desc": f'Large clear legible text reading "{display_text}" placed in the selected area.',
            }
            if color_palette:
                element["color_palette"] = color_palette[:5]
            elements.append(element)
        else:
            element = {
                "type": "obj",
                "bbox": bbox,
                "desc": desc or "Object or subject edit for the selected area.",
            }
            if color_palette:
                element["color_palette"] = color_palette[:5]
            elements.append(element)

    if not elements:
        return request.prompt

    video_config = request.video or {}
    style = video_config.get("ideogram_style") if isinstance(video_config, dict) else {}
    style = style if isinstance(style, dict) else {}
    background = str(video_config.get("ideogram_background") or side_prompt or "").strip()
    if not background:
        background = "Preserve the source image background and composition." if request.activity.lower() == "img2img" else "Scene background consistent with the high level description."
    prompt = {
        "high_level_description": _ideogram4_join_prompt_parts(side_prompt or "Natural final image with regional edits applied invisibly.", clean_directive),
        "style_description": {
            "aesthetics": str(style.get("aesthetics") or "clean, coherent, high detail"),
            "lighting": str(style.get("lighting") or "natural balanced lighting"),
            "photo": str(style.get("photo") or "high quality image with clear composition"),
            "medium": str(style.get("medium") or "photograph"),
        },
        "compositional_deconstruction": {
            "background": background,
            "elements": elements,
        },
    }
    return json.dumps(prompt, ensure_ascii=False, indent=2)


def _ideogram4_structured_prompt_is_full_scene(caption: dict[str, Any]) -> bool:
    comp = caption.get("compositional_deconstruction")
    elements = comp.get("elements") if isinstance(comp, dict) else []
    if not isinstance(elements, list) or len(elements) != 1 or not isinstance(elements[0], dict):
        return False
    element = elements[0]
    bbox = element.get("bbox")
    return (
        str(element.get("type") or "obj").lower() != "text"
        and not str(element.get("text") or "").strip()
        and isinstance(bbox, list)
        and len(bbox) == 4
        and [int(value) for value in bbox] == [0, 0, 1000, 1000]
    )


def _ideogram4_flatten_structured_scene_prompt(caption: dict[str, Any]) -> str:
    comp = caption.get("compositional_deconstruction") if isinstance(caption, dict) else {}
    comp = comp if isinstance(comp, dict) else {}
    style = caption.get("style_description") if isinstance(caption, dict) else {}
    style = style if isinstance(style, dict) else {}
    elements = comp.get("elements") if isinstance(comp, dict) else []
    element_desc = ""
    if isinstance(elements, list) and elements and isinstance(elements[0], dict):
        element_desc = str(elements[0].get("desc") or "").strip()
        element_desc = re.sub(
            r"^\s*unified\s+full-canvas\s+wide\s+action\s+scene\s+with\s+the\s+main\s+subjects\s+visible\s+and\s+interacting\s+in\s+one\s+shared\s+environment\s*:\s*",
            "",
            element_desc,
            flags=re.IGNORECASE,
        ).strip()
    parts = [
        str(caption.get("high_level_description") or "").strip(),
        element_desc,
        str(comp.get("background") or "").strip(),
        str(style.get("aesthetics") or "").strip(),
        str(style.get("lighting") or "").strip(),
        str(style.get("photo") or "").strip(),
        str(style.get("medium") or "").strip(),
    ]
    prompt = _ideogram4_join_prompt_parts(*parts)
    return _ideogram4_join_prompt_parts(
        "Single coherent scene, all requested subjects clearly visible in one shared camera view, natural camera distance, readable composition, no collage, no split-screen, no cropped close-up.",
        prompt,
    )


def _ideogram4_elements_data(request: GenerateRequest) -> str:
    regions = (request.video or {}).get("ideogram_regions")
    side_prompt = _ideogram4_effective_side_prompt(request)
    preserve_image = side_prompt.lower() == "preserve image"

    def _region_unit(value: float | None, default: float = 0.0) -> float:
        if value is None:
            return default
        number = float(value)
        if number > 1.0:
            number /= 100.0
        return max(0.0, min(1.0, number))

    elements: list[dict[str, Any]] = []
    if side_prompt and not preserve_image:
        elements.append(
            {
                "type": "obj",
                "text": "",
                "desc": _ideogram4_scene_element_desc(side_prompt),
                "palette": [],
                "x": 0.0,
                "y": 0.0,
                "w": 1.0,
                "h": 1.0,
            }
        )
    if not isinstance(regions, list):
        return json.dumps(elements, ensure_ascii=False, separators=(",", ":")) if elements else ""

    for item in regions[:24]:
        if not isinstance(item, dict):
            continue
        raw_x = _number_or_none(item.get("x"))
        raw_y = _number_or_none(item.get("y"))
        raw_w = _number_or_none(item.get("w"))
        raw_h = _number_or_none(item.get("h"))
        if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
            continue
        x = _region_unit(raw_x)
        y = _region_unit(raw_y)
        w = max(0.01, min(1.0 - x, _region_unit(raw_w, 0.25)))
        h = max(0.01, min(1.0 - y, _region_unit(raw_h, 0.25)))
        desc = str(item.get("prompt") or item.get("desc") or "").strip()
        text = str(item.get("text") or "").strip()
        element_type = str(item.get("type") or "obj").strip().lower()
        colors = item.get("colors")
        palette = [str(color).strip().upper() for color in colors if str(color).strip()] if isinstance(colors, list) else []
        if not palette and item.get("color"):
            palette = [str(item.get("color")).strip().upper()]
        if element_type == "text" or text:
            elements.append(
                {
                    "type": "text",
                    "text": text or desc,
                    "desc": "",
                    "palette": palette[:5],
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "w": round(w, 4),
                    "h": round(h, 4),
                }
            )
        else:
            elements.append(
                {
                    "type": "obj",
                    "text": "",
                    "desc": desc,
                    "palette": palette[:5],
                    "x": round(x, 4),
                    "y": round(y, 4),
                    "w": round(w, 4),
                    "h": round(h, 4),
                }
            )
    return json.dumps(elements, ensure_ascii=False, separators=(",", ":")) if elements else ""


def build_basic_ideogram4_workflow(
    request: GenerateRequest,
    model_name: str,
    unconditional_model_name: str,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str | None = None,
    available_nodes: set[str] | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    width = max(256, int(request.width))
    height = max(256, int(request.height))
    width = max(256, ((width + 15) // 16) * 16)
    height = max(256, ((height + 15) // 16) * 16)
    steps = max(1, int(request.steps or 12))
    cfg = float(request.cfg or 1.0)
    effective_side_prompt = _ideogram4_effective_side_prompt(request)
    clean_directive = _ideogram4_clean_output_directive(request)
    prompt_text = _ideogram4_regions_prompt(request)
    elements_data = _ideogram4_elements_data(request)
    regions = (request.video or {}).get("ideogram_regions") if isinstance(request.video, dict) else None
    structured_prompt = parse_ideogram4_prompt_json(request.prompt)
    prompt_is_structured_json = structured_prompt is not None
    use_prompt_builder = prompt_is_structured_json or bool(request.prompt.strip()) or (isinstance(regions, list) and bool(regions))
    effective_denoise = float((request.img2img.denoise if reference_image_name else request.denoise) or 1.0)
    if request.activity.lower() == "txt2img" and not reference_image_name:
        effective_denoise = 1.0
    available_nodes = set(available_nodes or ())
    noise_math_node = (
        "mrmth_ag_NoiseMathNode"
        if "mrmth_ag_NoiseMathNode" in available_nodes
        else ("mrmth_NoiseMathNode" if "mrmth_NoiseMathNode" in available_nodes else "")
    )
    use_method2 = bool(noise_math_node and "SamplerLCMCustom" in available_nodes)
    method2_noise_sampler = str((request.video or {}).get("ideogram_method2_noise_sampler") or "pyramid").strip() or "pyramid"

    model_ref: list[Any] = ["1", 0]
    lora_nodes: dict[str, Any] = {}
    next_lora_id = 30
    for lora_name, strength_model, _strength_clip in _active_lora_selections(request):
        node_id = str(next_lora_id)
        lora_nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": lora_name,
                "strength_model": float(strength_model),
            },
            "_meta": {"title": f"Ideogram 4 LoRA - {Path(lora_name).name}"},
        }
        model_ref = [node_id, 0]
        next_lora_id += 1

    latent_ref: list[Any] = ["7", 0]
    workflow: dict[str, Any] = {
        "1": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Load Ideogram 4 Model"},
        },
        "2": {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": unconditional_model_name, "weight_dtype": "default"},
            "_meta": {"title": "Load Ideogram 4 Unconditional Model"},
        },
        "3": _clip_loader_node(text_encoder_name, "ideogram4", "Ideogram 4 Qwen3-VL Text Encoder"),
        "4": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Flux2 VAE"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 0], "text": prompt_text},
            "_meta": {"title": "Ideogram 4 Prompt Encode"},
        },
        "6": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["5", 0]},
            "_meta": {"title": "Asymmetric Empty Negative"},
        },
        "7": {
            "class_type": "EmptyFlux2LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": max(1, request.batch_size)},
            "_meta": {"title": "Ideogram 4 Latent"},
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
            "_meta": {"title": "Noise"},
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": normalize_sampler(request.sampler or "euler")},
            "_meta": {"title": "Sampler"},
        },
        "10": {
            "class_type": "ModelSamplingAuraFlow",
            "inputs": {"model": model_ref, "shift": 7.0},
            "_meta": {"title": "Ideogram 4 AuraFlow Sampling"},
        },
        "11": {
            "class_type": "CFGOverride",
            "inputs": {"model": ["10", 0], "cfg": cfg, "start_percent": 0.0, "end_percent": 1.0},
            "_meta": {"title": "Ideogram 4 CFG Override"},
        },
        "12": {
            "class_type": "DualModelGuider",
            "inputs": {"model": ["11", 0], "positive": ["5", 0], "model_negative": ["2", 0], "negative": ["6", 0], "cfg": 7.0},
            "_meta": {"title": "Ideogram 4 Dual Model Guider"},
        },
        "13": (
            {
                "class_type": "BasicScheduler",
                "inputs": {"model": ["10", 0], "scheduler": normalize_scheduler(request.scheduler or "simple"), "steps": steps, "denoise": effective_denoise},
                "_meta": {"title": "Ideogram 4 Img2Img Scheduler"},
            }
            if reference_image_name
            else {
                "class_type": "Ideogram4Scheduler",
                "inputs": {"steps": steps, "width": width, "height": height, "mu": 0.5, "std": 1.75},
                "_meta": {"title": "Ideogram 4 Scheduler"},
            }
        ),
        "14": {
            "class_type": "ExtendIntermediateSigmas",
            "inputs": {"sigmas": ["13", 0], "steps": 2, "start_at_sigma": 1.0, "end_at_sigma": 0.98, "spacing": "linear"},
            "_meta": {"title": "Ideogram 4 KJ Sigma Extension"},
        },
        "15": {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["8", 0], "guider": ["12", 0], "sampler": ["9", 0], "sigmas": ["14", 0], "latent_image": ["7", 0]},
            "_meta": {"title": "Ideogram 4 Sampler"},
        },
        "16": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["15", 0], "vae": ["4", 0]},
            "_meta": {"title": "VAE Decode"},
        },
        "17": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "NEXUS_BTA_IDEOGRAM4", "images": ["16", 0]},
            "_meta": {"title": "Save Image"},
        },
    }
    if use_prompt_builder:
        import_json = ""
        import_mode = "when empty"
        builder_high_level = _ideogram4_join_prompt_parts(effective_side_prompt, clean_directive)
        builder_background = _ideogram4_join_prompt_parts(
            effective_side_prompt or ("Preserve the source image background and composition." if request.activity.lower() == "img2img" else "A coherent high quality scene."),
        )
        builder_style_photo = "high resolution, clear photo, natural surfaces, coherent subjects"
        builder_aesthetics = "clean, detailed, natural, edge-to-edge image"
        builder_lighting = "natural balanced lighting"
        builder_medium = "photograph"
        builder_elements_data = elements_data
        if structured_prompt is not None:
            import_json = ideogram4_prompt_json_text(structured_prompt)
            import_mode = "always"
            builder_elements_data = ""
        workflow["5"] = {
            "class_type": "Ideogram4PromptBuilderKJ",
            "inputs": {
                "width": width,
                "height": height,
                "high_level_description": builder_high_level,
                "background": builder_background,
                "style": "photo",
                "style.photo": builder_style_photo,
                "aesthetics": builder_aesthetics,
                "lighting": builder_lighting,
                "medium": builder_medium,
                "import_json": import_json,
                "import_mode": import_mode,
                "style_palette_data": "",
                "elements_data": builder_elements_data,
                "bg_brightness": 25,
            },
            "_meta": {"title": "Ideogram 4 KJ Prompt Builder"},
        }
        workflow["21"] = {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["3", 0], "text": ["5", 0]},
            "_meta": {"title": "Ideogram 4 Structured Prompt Encode"},
        }
        workflow["6"]["inputs"]["conditioning"] = ["21", 0]
        workflow["12"]["inputs"]["positive"] = ["21", 0]
    if reference_image_name:
        workflow["18"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
            "_meta": {"title": "Ideogram 4 Base Reference"},
        }
        workflow["19"] = {
            "class_type": "ImageScale",
            "inputs": {
                "image": ["18", 0],
                "upscale_method": "lanczos",
                "width": width,
                "height": height,
                "crop": "center",
            },
            "_meta": {"title": "Scale Reference To Side Menu Resolution"},
        }
        workflow["20"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": ["19", 0], "vae": ["4", 0]},
            "_meta": {"title": "Ideogram 4 Img2Img Latent"},
        }
        latent_ref = ["20", 0]
        workflow["15"]["inputs"]["latent_image"] = latent_ref
        if use_prompt_builder:
            workflow["5"]["inputs"]["image"] = ["19", 0]
    if use_method2:
        method2_noise_expr = "a*2" if not reference_image_name else "a"
        noise_math_inputs = (
            {"V.V0": ["8", 0], "Noise": method2_noise_expr, "remember_stack": False}
            if noise_math_node == "mrmth_ag_NoiseMathNode"
            else {"a": ["8", 0], "Noise": method2_noise_expr}
        )
        method2_split_step = int((request.video or {}).get("ideogram_method2_split_step") or 1)
        workflow["90"] = {
            "class_type": noise_math_node,
            "inputs": noise_math_inputs,
            "_meta": {"title": "Ideogram 4 Method 2 Initial Noise"},
        }
        workflow["91"] = {
            "class_type": "SamplerLCMCustom",
            "inputs": {"noise_sampler_type": method2_noise_sampler},
            "_meta": {"title": "Ideogram 4 Method 2 LCM Sampler"},
        }
        workflow["92"] = {
            "class_type": "SplitSigmas",
            "inputs": {"sigmas": ["13", 0], "step": max(1, min(3, method2_split_step))},
            "_meta": {"title": "Ideogram 4 Method 2 Sigma Split"},
        }
        workflow["93"] = {
            "class_type": "SamplerCustomAdvanced",
            "inputs": {"noise": ["90", 0], "guider": ["12", 0], "sampler": ["91", 0], "sigmas": ["92", 0], "latent_image": latent_ref},
            "_meta": {"title": "Ideogram 4 Method 2 High Sigma Pass"},
        }
        workflow["94"] = {
            "class_type": "DisableNoise",
            "inputs": {},
            "_meta": {"title": "No Extra Noise For Low Sigma Pass"},
        }
        workflow["15"]["inputs"]["noise"] = ["94", 0]
        workflow["15"]["inputs"]["sigmas"] = ["92", 1]
        workflow["15"]["inputs"]["latent_image"] = ["93", 0]
        workflow.pop("14", None)
    workflow.update(lora_nodes)
    return workflow


def build_basic_flux_workflow(
    request: GenerateRequest,
    model_name: str,
    clip_l_name: str | None,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str | None = None,
    reference_image_names: list[str] | None = None,
    mask_image_name: str | None = None,
    flux_family: str | None = None,
    controlnet_name: str | None = None,
    controlnet_image_name: str | None = None,
    available_nodes: set[str] | None = None,
) -> dict[str, Any]:
    _apply_outpaint_continuity_prompt(request)
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    width = max(16, int(request.width))
    height = max(16, int(request.height))
    available_nodes = set(available_nodes or ())
    sampler = normalize_sampler(request.sampler or "euler")
    scheduler = normalize_scheduler(request.scheduler or "simple")
    flux_family = flux_family or _flux_family_from_name(model_name)
    is_flux2 = flux_family.startswith("flux2")
    is_klein = "klein" in flux_family
    flux_multiview = _flux_multiview_options(request)
    loader = (
        {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": model_name},
            "_meta": {"title": "Load Flux GGUF Model"},
        }
        if model_name.lower().endswith(".gguf")
        else {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Load Flux Model"},
        }
    )
    flux_guidance = max(0.0, float(request.cfg if request.cfg is not None else 3.5))
    latent_ref: list[Any] = ["6", 0]
    denoise = request.img2img.denoise if reference_image_name else request.denoise
    flux2_reference_names: list[str] = []
    for name in (reference_image_names or ([reference_image_name] if reference_image_name else [])):
        normalized = str(name or "").strip()
        if normalized and normalized not in flux2_reference_names:
            flux2_reference_names.append(normalized)
    if is_flux2:
        mode_lower = str(request.img2img.mode or "").lower()
        flux2_outpaint_mode = _uses_inpaint_mask_mode(request) and ("outpaint" in mode_lower or "extend" in mode_lower)
        positive_ref: list[Any] = ["4", 0]
        workflow = {
            "1": loader,
            "2": _clip_loader_node(text_encoder_name, "flux2", "Flux.2 Text Encoder"),
            "3": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": vae_name},
                "_meta": {"title": "Flux.2 VAE"},
            },
            "4": {
                "class_type": "CLIPTextEncode",
                "inputs": {"text": request.prompt, "clip": ["2", 0]},
                "_meta": {"title": "Positive Prompt"},
            },
            "5": {
                "class_type": "ConditioningZeroOut",
                "inputs": {"conditioning": ["4", 0]},
                "_meta": {"title": "Flux.2 Empty Negative"},
            },
            "6": {
                "class_type": "EmptyFlux2LatentImage",
                "inputs": {"width": width, "height": height, "batch_size": max(1, request.batch_size)},
                "_meta": {"title": "Flux.2 Latent"},
            },
            "7": {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed},
                "_meta": {"title": "Seed"},
            },
            "8": {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": sampler},
                "_meta": {"title": "Sampler"},
            },
            "9": {
                "class_type": "Flux2Scheduler",
                "inputs": {"steps": max(1, request.steps), "width": width, "height": height},
                "_meta": {"title": "Flux.2 Scheduler"},
            },
            "10": {
                "class_type": "BasicGuider",
                "inputs": {"model": ["1", 0], "conditioning": positive_ref},
                "_meta": {"title": "Flux.2 Guider"},
            },
            "11": {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {"noise": ["7", 0], "guider": ["10", 0], "sampler": ["8", 0], "sigmas": ["9", 0], "latent_image": latent_ref},
                "_meta": {"title": "Flux.2 Sampler"},
            },
            "12": {
                "class_type": "VAEDecode",
                "inputs": {"samples": ["11", 0], "vae": ["3", 0]},
                "_meta": {"title": "VAE Decode"},
            },
            "15": _image_scale_node(["12", 0], int(request.width), int(request.height)),
            "13": {
                "class_type": "SaveImage",
                "inputs": {"filename_prefix": "NEXUS_BTA_FLUX2", "images": ["15", 0]},
                "_meta": {"title": "Save Image"},
            },
        }
        workflow["15"]["_meta"]["title"] = "Resize Flux.2 Output To Frontend Size"
        if not is_klein:
            workflow["14"] = {
                "class_type": "FluxGuidance",
                "inputs": {"conditioning": ["4", 0], "guidance": flux_guidance},
                "_meta": {"title": "Flux.2 Guidance"},
            }
            positive_ref = ["14", 0]
            workflow["10"]["inputs"]["conditioning"] = positive_ref
        model_ref, _, _ = _append_lora_chain(workflow, request, ["1", 0], ["2", 0], start_id=20, model_only=True)
        workflow["10"]["inputs"]["model"] = model_ref
        if flux2_reference_names:
            previous_conditioning = positive_ref
            base_loader_id: str | None = None
            base_encode_id: str | None = None
            base_mask_ref: list[Any] | None = None
            for index, name in enumerate(flux2_reference_names[:5], start=1):
                load_id = str(40 + (index - 1) * 4)
                scale_id = str(41 + (index - 1) * 4)
                encode_id = str(42 + (index - 1) * 4)
                ref_id = str(43 + (index - 1) * 4)
                workflow[load_id] = {
                    "class_type": "LoadImage",
                    "inputs": {"image": name},
                    "_meta": {"title": f"Flux.2 Reference Image {index}"},
                }
                if flux2_outpaint_mode and index == 1 and _has_extend_pad(request):
                    if _uses_prepadded_outpaint_reference(request):
                        workflow[scale_id] = _image_scale_node([load_id, 0], width, height)
                        workflow[scale_id]["_meta"]["title"] = "Flux.2 Pre-Padded Extend Reference"
                    else:
                        workflow[scale_id] = _image_pad_for_outpaint_node([load_id, 0], request, feathering=40, available_nodes=available_nodes)
                        workflow[scale_id]["_meta"]["title"] = "Flux.2 Pad Base For Extend"
                        base_mask_ref = [scale_id, 1]
                else:
                    workflow[scale_id] = _image_scale_node([load_id, 0], width, height)
                    workflow[scale_id]["_meta"]["title"] = f"Resize Flux.2 Reference {index} To Side Menu"
                workflow[encode_id] = {
                    "class_type": "VAEEncode",
                    "inputs": {"pixels": [scale_id, 0], "vae": ["3", 0]},
                    "_meta": {"title": f"Encode Flux.2 Reference {index}"},
                }
                workflow[ref_id] = {
                    "class_type": "ReferenceLatent",
                    "inputs": {"conditioning": previous_conditioning, "latent": [encode_id, 0]},
                    "_meta": {"title": f"Flux.2 Reference Latent {index}"},
                }
                previous_conditioning = [ref_id, 0]
                if index == 1:
                    base_loader_id = scale_id
                    base_encode_id = encode_id
                    if flux_multiview["enabled"]:
                        workflow["90"] = {
                            "class_type": "QwenMultiangleCameraNode",
                            "inputs": {
                                "image": [scale_id, 0],
                                "horizontal_angle": int(round(float(flux_multiview["horizontal"]))),
                                "vertical_angle": int(round(float(flux_multiview["vertical"]))),
                                "zoom": float(flux_multiview["zoom"]),
                                "default_prompts": True,
                                "camera_view": bool(flux_multiview["camera_view"]),
                            },
                            "_meta": {"title": "Flux Multiangle Camera"},
                        }
                        workflow["4"]["inputs"]["text"] = ["90", 0]
            if previous_conditioning != positive_ref:
                workflow["10"]["inputs"]["conditioning"] = previous_conditioning
            if base_encode_id:
                workflow["11"]["inputs"]["latent_image"] = [base_encode_id, 0]
            if base_loader_id:
                _append_inpaint_mask(
                    workflow,
                    request,
                    reference_node_id=base_loader_id,
                    vae_ref=["3", 0],
                    sampler_node_id="11",
                    mask_image_name=mask_image_name,
                    mask_ref_override=base_mask_ref,
                    start_id=80,
                    decoded_image_ref=["15", 0],
                    save_node_id="13",
                    available_nodes=available_nodes,
                )
        positive_ref, _, _ = _append_flux_union_controlnet_route(
            workflow,
            request,
            positive_ref=list(workflow["10"]["inputs"]["conditioning"]),
            negative_ref=["5", 0],
            controlnet_name=controlnet_name,
            controlnet_image_name=controlnet_image_name,
            vae_ref=["3", 0],
            start_id=120,
        )
        workflow["10"]["inputs"]["conditioning"] = positive_ref
        return workflow

    workflow = {
        "1": loader,
        "2": {
            "class_type": "DualCLIPLoader",
            "inputs": {
                "clip_name1": clip_l_name,
                "clip_name2": text_encoder_name,
                "type": "flux",
                "device": "default",
            },
            "_meta": {"title": "Flux CLIP-L + T5"},
        },
        "3": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Flux VAE"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": request.prompt, "clip": ["2", 0]},
            "_meta": {"title": "Positive Prompt"},
        },
        "5": {
            "class_type": "ConditioningZeroOut",
            "inputs": {"conditioning": ["4", 0]},
            "_meta": {"title": "Flux Empty Negative"},
        },
        "12": {
            "class_type": "FluxGuidance",
            "inputs": {"conditioning": ["4", 0], "guidance": flux_guidance},
            "_meta": {"title": "Flux Guidance"},
        },
        "6": {
            "class_type": "EmptySD3LatentImage",
            "inputs": {"width": width, "height": height, "batch_size": max(1, request.batch_size)},
            "_meta": {"title": "Flux Latent"},
        },
        "7": {
            "class_type": "KSampler",
            "inputs": {
                "seed": seed,
                "steps": max(1, request.steps),
                "cfg": 1.0,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "denoise": denoise,
                "model": ["1", 0],
                "positive": ["12", 0],
                "negative": ["5", 0],
                "latent_image": latent_ref,
            },
            "_meta": {"title": "Flux Sampler"},
        },
        "8": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["7", 0], "vae": ["3", 0]},
            "_meta": {"title": "VAE Decode"},
        },
        "14": _image_scale_node(["8", 0], int(request.width), int(request.height)),
        "9": {
            "class_type": "SaveImage",
            "inputs": {"filename_prefix": "NEXUS_BTA_FLUX", "images": ["14", 0]},
            "_meta": {"title": "Save Image"},
        },
    }
    workflow["14"]["_meta"]["title"] = "Resize Flux Output To Frontend Size"
    model_ref, clip_ref, _ = _append_lora_chain(workflow, request, ["1", 0], ["2", 0], start_id=20)
    workflow["4"]["inputs"]["clip"] = clip_ref
    workflow["7"]["inputs"]["model"] = model_ref
    if reference_image_name:
        workflow["10"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
            "_meta": {"title": "Reference Image"},
        }
        if _uses_outpaint_extend_mode(request) and _has_extend_pad(request):
            if _uses_prepadded_outpaint_reference(request):
                workflow["13"] = _image_scale_node(["10", 0], width, height)
                workflow["13"]["_meta"]["title"] = "Flux Pre-Padded Extend Reference"
            else:
                workflow["13"] = _image_pad_for_outpaint_node(["10", 0], request, feathering=40, available_nodes=available_nodes)
                workflow["13"]["_meta"]["title"] = "Flux Pad Base For Extend"
            reference_encode_image_ref = ["13", 0]
        else:
            workflow["13"] = _image_scale_node(["10", 0], width, height)
            reference_encode_image_ref = ["13", 0]
        workflow["11"] = {
            "class_type": "VAEEncode",
            "inputs": {"pixels": reference_encode_image_ref, "vae": ["3", 0]},
            "_meta": {"title": "Encode Reference"},
        }
        workflow["7"]["inputs"]["latent_image"] = ["11", 0]
        _append_inpaint_mask(
            workflow,
            request,
            reference_node_id="13",
            vae_ref=["3", 0],
            sampler_node_id="7",
            mask_image_name=mask_image_name,
            decoded_image_ref=["14", 0],
            save_node_id="9",
            available_nodes=available_nodes,
        )
    positive_ref, negative_ref, _ = _append_flux_union_controlnet_route(
        workflow,
        request,
        positive_ref=list(workflow["7"]["inputs"]["positive"]),
        negative_ref=list(workflow["7"]["inputs"]["negative"]),
        controlnet_name=controlnet_name,
        controlnet_image_name=controlnet_image_name,
        vae_ref=["3", 0],
        start_id=120,
    )
    workflow["7"]["inputs"]["positive"] = positive_ref
    workflow["7"]["inputs"]["negative"] = negative_ref
    return workflow


def build_basic_wan_i2video_workflow(
    request: GenerateRequest,
    high_model_name: str,
    low_model_name: str,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str | None = None,
    reference_end_image_name: str | None = None,
    first_last_frame_node: str | None = None,
    clip_vision_name: str | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    video_options = request.video or {}
    loop_cycle = _truthy_option(video_options.get("wan_loop_cycle"))
    fps = max(1, int(_number_or_none(video_options.get("fps")) or 16))
    seconds = _number_or_none(video_options.get("seconds") or video_options.get("duration"))
    requested_frames = _number_or_none(video_options.get("frames") or video_options.get("length"))
    if requested_frames:
        length = max(5, int(round(requested_frames)))
    elif seconds:
        length = max(5, int(round(seconds * fps)) + 1)
    else:
        length = 81
    if (length - 1) % 4 != 0:
        length = (((length - 1) // 4) + 1) * 4 + 1

    width = max(64, int(request.width))
    height = max(64, int(request.height))
    width -= width % 16
    height -= height % 16

    steps = 4
    cfg = float(request.cfg if request.cfg is not None else 1.0)
    sampler = normalize_sampler(request.sampler or "euler")
    scheduler = normalize_scheduler(request.scheduler or "simple")
    positive_prompt = (request.prompt or "").strip()
    negative_prompt = (request.negative_prompt or "").strip()
    if loop_cycle:
        loop_positive = "seamless perfect looping video, cyclic motion, first frame matches last frame, no hard cut at loop point"
        loop_negative = "jump cut, hard cut, scene change, flicker at loop seam, black frame"
        positive_prompt = f"{positive_prompt}, {loop_positive}" if positive_prompt else loop_positive
        negative_prompt = f"{negative_prompt}, {loop_negative}" if negative_prompt else loop_negative
    split_step = max(1, min(steps - 1, steps // 2))
    high_model_ref: list[Any] = ["1", 0]
    low_model_ref: list[Any] = ["2", 0]
    wan_lora_nodes: dict[str, Any] = {}
    next_lora_id = 20

    def wan_loader(model_name: str) -> dict[str, Any]:
        if model_name.lower().endswith(".gguf"):
            return {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": model_name},
                "_meta": {"title": "Load WAN GGUF Model"},
            }
        return {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Load WAN Model"},
        }

    clip_loader = (
        {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": text_encoder_name, "type": "wan"},
            "_meta": {"title": "WAN UMT5 Encoder"},
        }
        if text_encoder_name.lower().endswith(".gguf")
        else {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder_name, "type": "wan", "device": "default"},
            "_meta": {"title": "WAN UMT5 Encoder"},
        }
    )

    for lora_name, strength_model, _strength_clip in _active_lora_selections(request):
        safe_strength = max(-2.0, min(2.0, float(strength_model)))
        lora_lower = lora_name.lower()
        apply_high = "low" not in lora_lower or "high" in lora_lower
        apply_low = "high" not in lora_lower or "low" in lora_lower
        if apply_high:
            node_id = str(next_lora_id)
            wan_lora_nodes[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": high_model_ref, "lora_name": lora_name, "strength_model": safe_strength},
                "_meta": {"title": f"WAN high LoRA - {Path(lora_name).name}"},
            }
            high_model_ref = [node_id, 0]
            next_lora_id += 1
        if apply_low:
            node_id = str(next_lora_id)
            wan_lora_nodes[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": low_model_ref, "lora_name": lora_name, "strength_model": safe_strength},
                "_meta": {"title": f"WAN low LoRA - {Path(lora_name).name}"},
            }
            low_model_ref = [node_id, 0]
            next_lora_id += 1

    workflow = {
        "1": wan_loader(high_model_name),
        "2": wan_loader(low_model_name),
        "3": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": high_model_ref, "shift": float(video_options.get("shift") or 5.0)},
            "_meta": {"title": "WAN High Noise Shift"},
        },
        "4": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": low_model_ref, "shift": float(video_options.get("shift") or 5.0)},
            "_meta": {"title": "WAN Low Noise Shift"},
        },
        "5": clip_loader,
        "6": {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "WAN VAE"},
        },
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": positive_prompt},
            "_meta": {"title": "Positive Prompt"},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": negative_prompt},
            "_meta": {"title": "Negative Prompt"},
        },
        "10": {
            "class_type": first_last_frame_node or "WanImageToVideo",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "vae": ["6", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": max(1, request.batch_size),
            },
            "_meta": {"title": "WAN Image To Video"},
        },
        "11": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["3", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["10", 2],
                "start_at_step": 0,
                "end_at_step": split_step,
                "return_with_leftover_noise": "enable",
            },
            "_meta": {"title": "WAN High Noise Sampler"},
        },
        "12": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["4", 0],
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "positive": ["10", 0],
                "negative": ["10", 1],
                "latent_image": ["11", 0],
                "start_at_step": split_step,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable",
            },
            "_meta": {"title": "WAN Low Noise Sampler"},
        },
        "13": {
            "class_type": "VAEDecode",
            "inputs": {"samples": ["12", 0], "vae": ["6", 0]},
            "_meta": {"title": "Decode Frames"},
        },
        "14": {
            "class_type": "CreateVideo",
            "inputs": {"images": ["13", 0], "fps": float(fps)},
            "_meta": {"title": "Create Video"},
        },
        "15": {
            "class_type": "SaveVideo",
            "inputs": {
                "video": ["14", 0],
                "filename_prefix": "NEXUS_BTA_WAN22_LOOP_CYCLE" if loop_cycle else "NEXUS_BTA_WAN22_I2V_512",
                "format": "mp4",
                "codec": "h264",
            },
            "_meta": {"title": "Save Video"},
        },
    }

    if reference_image_name:
        workflow["9"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
            "_meta": {"title": "Reference Image"},
        }
        workflow["10"]["inputs"]["start_image"] = ["9", 0]
    if reference_image_name and reference_end_image_name and first_last_frame_node:
        workflow["16"] = {
            "class_type": "LoadImage",
            "inputs": {"image": reference_end_image_name},
            "_meta": {"title": "End Frame Image"},
        }
        workflow["10"]["inputs"]["end_image"] = ["16", 0]
        if clip_vision_name:
            workflow["17"] = {
                "class_type": "CLIPVisionLoader",
                "inputs": {"clip_name": clip_vision_name},
                "_meta": {"title": "WAN CLIP Vision"},
            }
            workflow["18"] = {
                "class_type": "CLIPVisionEncode",
                "inputs": {"clip_vision": ["17", 0], "image": ["9", 0], "crop": "center"},
                "_meta": {"title": "Encode WAN Start Frame Vision"},
            }
            workflow["19"] = {
                "class_type": "CLIPVisionEncode",
                "inputs": {"clip_vision": ["17", 0], "image": ["16", 0], "crop": "center"},
                "_meta": {"title": "Encode WAN End Frame Vision"},
            }
            workflow["10"]["inputs"]["clip_vision_start_image"] = ["18", 0]
            workflow["10"]["inputs"]["clip_vision_end_image"] = ["19", 0]
    workflow.update(wan_lora_nodes)
    return workflow


def build_basic_wan_video_reference_workflow(
    request: GenerateRequest,
    high_model_name: str,
    low_model_name: str,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str,
    base_video_name: str,
    clip_vision_name: str | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    video_options = request.video or {}
    fps = max(1, int(_number_or_none(video_options.get("fps")) or 16))
    seconds = _number_or_none(video_options.get("seconds") or video_options.get("duration"))
    requested_frames = _number_or_none(video_options.get("frames") or video_options.get("length"))
    if requested_frames:
        length = max(5, int(round(requested_frames)))
    elif seconds:
        length = max(5, int(round(seconds * fps)) + 1)
    else:
        length = 81
    if (length - 1) % 4 != 0:
        length = (((length - 1) // 4) + 1) * 4 + 1

    width = max(64, int(request.width))
    height = max(64, int(request.height))
    width -= width % 16
    height -= height % 16

    steps = 4
    cfg = float(request.cfg if request.cfg is not None else 1.0)
    sampler = normalize_sampler(request.sampler or "euler")
    scheduler = normalize_scheduler(request.scheduler or "simple")
    split_step = max(1, min(steps - 1, steps // 2))
    high_model_ref: list[Any] = ["1", 0]
    low_model_ref: list[Any] = ["2", 0]
    wan_lora_nodes: dict[str, Any] = {}
    next_lora_id = 30

    def wan_loader(model_name: str) -> dict[str, Any]:
        if model_name.lower().endswith(".gguf"):
            return {
                "class_type": "UnetLoaderGGUF",
                "inputs": {"unet_name": model_name},
                "_meta": {"title": "Load WAN GGUF Model"},
            }
        return {
            "class_type": "UNETLoader",
            "inputs": {"unet_name": model_name, "weight_dtype": "default"},
            "_meta": {"title": "Load WAN Model"},
        }

    clip_loader = (
        {
            "class_type": "CLIPLoaderGGUF",
            "inputs": {"clip_name": text_encoder_name, "type": "wan"},
            "_meta": {"title": "WAN UMT5 Encoder"},
        }
        if text_encoder_name.lower().endswith(".gguf")
        else {
            "class_type": "CLIPLoader",
            "inputs": {"clip_name": text_encoder_name, "type": "wan", "device": "default"},
            "_meta": {"title": "WAN UMT5 Encoder"},
        }
    )

    for lora_name, strength_model, _strength_clip in _active_lora_selections(request):
        safe_strength = max(-2.0, min(2.0, float(strength_model)))
        lora_lower = lora_name.lower()
        apply_high = "low" not in lora_lower or "high" in lora_lower
        apply_low = "high" not in lora_lower or "low" in lora_lower
        if apply_high:
            node_id = str(next_lora_id)
            wan_lora_nodes[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": high_model_ref, "lora_name": lora_name, "strength_model": safe_strength},
                "_meta": {"title": f"WAN high LoRA - {Path(lora_name).name}"},
            }
            high_model_ref = [node_id, 0]
            next_lora_id += 1
        if apply_low:
            node_id = str(next_lora_id)
            wan_lora_nodes[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {"model": low_model_ref, "lora_name": lora_name, "strength_model": safe_strength},
                "_meta": {"title": f"WAN low LoRA - {Path(lora_name).name}"},
            }
            low_model_ref = [node_id, 0]
            next_lora_id += 1

    workflow = {
        "1": wan_loader(high_model_name),
        "2": wan_loader(low_model_name),
        "3": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": high_model_ref, "shift": float(video_options.get("shift") or 5.0)},
            "_meta": {"title": "WAN High Noise Shift"},
        },
        "4": {
            "class_type": "ModelSamplingSD3",
            "inputs": {"model": low_model_ref, "shift": float(video_options.get("shift") or 5.0)},
            "_meta": {"title": "WAN Low Noise Shift"},
        },
        "5": clip_loader,
        "6": {"class_type": "VAELoader", "inputs": {"vae_name": vae_name}, "_meta": {"title": "WAN VAE"}},
        "7": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": request.prompt},
            "_meta": {"title": "Positive Prompt"},
        },
        "8": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["5", 0], "text": request.negative_prompt},
            "_meta": {"title": "Negative Prompt"},
        },
        "9": {"class_type": "LoadImage", "inputs": {"image": reference_image_name}, "_meta": {"title": "Reference Character"}},
        "10": {
            "class_type": "VHS_LoadVideo",
            "inputs": {
                "video": base_video_name,
                "force_rate": float(fps),
                "custom_width": width,
                "custom_height": height,
                "frame_load_cap": length,
                "skip_first_frames": 0,
                "select_every_nth": 1,
                "format": "Wan",
            },
            "_meta": {"title": "Base Motion Video"},
        },
        "11": {
            "class_type": "WanAnimateToVideo",
            "inputs": {
                "positive": ["7", 0],
                "negative": ["8", 0],
                "vae": ["6", 0],
                "width": width,
                "height": height,
                "length": length,
                "batch_size": max(1, request.batch_size),
                "continue_motion_max_frames": min(length, 5),
                "video_frame_offset": 0,
                "reference_image": ["9", 0],
                "pose_video": ["10", 0],
                "continue_motion": ["10", 0],
            },
            "_meta": {"title": "WAN Video Motion Reference"},
        },
        "12": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["3", 0],
                "add_noise": "enable",
                "noise_seed": seed,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["11", 2],
                "start_at_step": 0,
                "end_at_step": split_step,
                "return_with_leftover_noise": "enable",
            },
            "_meta": {"title": "WAN High Noise Sampler"},
        },
        "13": {
            "class_type": "KSamplerAdvanced",
            "inputs": {
                "model": ["4", 0],
                "add_noise": "disable",
                "noise_seed": 0,
                "steps": steps,
                "cfg": cfg,
                "sampler_name": sampler,
                "scheduler": scheduler,
                "positive": ["11", 0],
                "negative": ["11", 1],
                "latent_image": ["12", 0],
                "start_at_step": split_step,
                "end_at_step": steps,
                "return_with_leftover_noise": "disable",
            },
            "_meta": {"title": "WAN Low Noise Sampler"},
        },
        "14": {"class_type": "VAEDecode", "inputs": {"samples": ["13", 0], "vae": ["6", 0]}, "_meta": {"title": "Decode Frames"}},
        "15": {"class_type": "CreateVideo", "inputs": {"images": ["14", 0], "fps": float(fps)}, "_meta": {"title": "Create Video"}},
        "16": {
            "class_type": "SaveVideo",
            "inputs": {"video": ["15", 0], "filename_prefix": "NEXUS_BTA_WAN22_V2V", "format": "mp4", "codec": "h264"},
            "_meta": {"title": "Save Video"},
        },
    }

    if clip_vision_name:
        workflow["17"] = {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": clip_vision_name}, "_meta": {"title": "WAN CLIP Vision"}}
        workflow["18"] = {
            "class_type": "CLIPVisionEncode",
            "inputs": {"clip_vision": ["17", 0], "image": ["9", 0], "crop": "center"},
            "_meta": {"title": "Encode Reference Character"},
        }
        workflow["11"]["inputs"]["clip_vision_output"] = ["18", 0]

    workflow.update(wan_lora_nodes)
    return workflow


def build_basic_wan_motion_capture_workflow(
    request: GenerateRequest,
    high_model_name: str,
    low_model_name: str,
    text_encoder_name: str,
    vae_name: str,
    reference_image_name: str,
    motion_video_name: str,
    clip_vision_name: str | None = None,
) -> dict[str, Any]:
    video_options = request.video or {}
    workflow = build_basic_wan_video_reference_workflow(
        request,
        high_model_name,
        low_model_name,
        text_encoder_name,
        vae_name,
        reference_image_name,
        motion_video_name,
        clip_vision_name=clip_vision_name,
    )
    width = max(64, int(request.width))
    height = max(64, int(request.height))
    width -= width % 16
    height -= height % 16

    workflow["10"]["_meta"]["title"] = "Motion Source Video"
    workflow["10"]["inputs"]["format"] = "AnimateDiff"
    workflow["11"] = {
        "class_type": "Wan22FunControlToVideo",
        "inputs": {
            "positive": ["7", 0],
            "negative": ["8", 0],
            "vae": ["6", 0],
            "width": width,
            "height": height,
            "length": int(workflow["11"]["inputs"]["length"]),
            "batch_size": max(1, request.batch_size),
            "ref_image": ["9", 0],
            "control_video": ["19", 0],
        },
        "_meta": {"title": "WAN 2.2 Motion Capture DWPose Control"},
    }
    pose_inputs: dict[str, Any] = {
        "image": ["10", 0],
        "detect_hand": "enable",
        "detect_body": "enable",
        "detect_face": "enable",
        "resolution": max(width, height),
        "bbox_detector": "yolox_l.onnx",
        "pose_estimator": "dw-ll_ucoco_384_bs5.torchscript.pt",
        "scale_stick_for_xinsr_cn": "disable",
    }
    workflow["19"] = {
        "class_type": "DWPreprocessor",
        "inputs": pose_inputs,
        "_meta": {"title": "WAN Motion DWPose Control Video"},
    }
    workflow.pop("17", None)
    workflow.pop("18", None)
    workflow["16"]["inputs"]["filename_prefix"] = "NEXUS_BTA_WAN22_MOTION_CAPTURE"
    workflow["16"]["_meta"]["title"] = "Save Motion Capture Video"
    return workflow


def _append_lora_chain(
    workflow: dict[str, Any],
    request: GenerateRequest,
    model_ref: list[Any],
    clip_ref: list[Any],
    *,
    start_id: int,
    model_only: bool = False,
) -> tuple[list[Any], list[Any], int]:
    next_id = start_id
    for lora_name, strength_model, strength_clip in _active_lora_selections(request, model_name=request.model_name or request.model_path):
        node_id = str(next_id)
        if model_only:
            workflow[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": model_ref,
                    "lora_name": lora_name,
                    "strength_model": strength_model,
                },
                "_meta": {"title": f"Flux.2 LoRA - {Path(lora_name).name}"},
            }
            model_ref = [node_id, 0]
        else:
            workflow[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": model_ref,
                    "clip": clip_ref,
                    "lora_name": lora_name,
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                },
                "_meta": {"title": f"LoRA - {Path(lora_name).name}"},
            }
            model_ref = [node_id, 0]
            clip_ref = [node_id, 1]
        next_id += 1
    return model_ref, clip_ref, next_id


def _active_lora_selections(request: GenerateRequest, model_name: str | None = None) -> list[tuple[str, float, float]]:
    selections: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    flux_family = _flux_family_from_name(model_name or request.model_name or request.model_path or "")

    def append_selection(name: str, strength_model: float, strength_clip: float = 0.0, *, allow_cross_folder: bool = False) -> None:
        normalized = _normalize_lora_name(name)
        if request.preset.lower() == "ltx" and "\\" not in normalized and normalized.lower().startswith(("ltx", "singularity")):
            normalized = f"ltx\\{normalized}"
        if not normalized:
            return
        if not allow_cross_folder and not _lora_is_compatible_with_preset(normalized, request.preset):
            return
        if request.preset.lower() == "flux" and not _flux_lora_is_compatible(normalized, flux_family):
            return
        key = normalized.lower()
        if key in seen:
            return
        seen.add(key)
        selections.append((normalized, float(strength_model), float(strength_clip)))

    def append_user_loras() -> None:
        for item in request.loras:
            if not isinstance(item, dict):
                continue
            raw_name = item.get("relative_name") or item.get("relative_path") or item.get("lora_name") or item.get("name")
            name = _normalize_lora_name(raw_name)
            if not name:
                continue
            if request.preset.lower() == "flux" and not _flux_lora_is_compatible(name, flux_family):
                continue
            strength_model = _number_or_none(item.get("strength_model", item.get("strength", 1.0))) or 1.0
            strength_clip_value = _number_or_none(item.get("strength_clip", item.get("clip_strength")))
            if request.preset.lower() in {"flux", "ltx", "qwen", "wan", "zimageturbo", "zimage"}:
                strength_clip = strength_clip_value if strength_clip_value is not None else 0.0
            else:
                strength_clip = strength_model if strength_clip_value in {None, 0.0} else strength_clip_value
            append_selection(name, float(strength_model), float(strength_clip), allow_cross_folder=True)

    def append_distilled_loras() -> None:
        for item in request.distilled_loras:
            raw_name = getattr(item, "name", "")
            name = _normalize_lora_name(raw_name)
            if not name or not _lora_is_compatible_with_preset(name, request.preset):
                continue
            if request.preset.lower() == "flux" and not _flux_lora_is_compatible(name, flux_family):
                continue
            if request.preset.lower() == "qwen" and request.activity == "img2img" and _is_incompatible_qwen_edit_lora(name):
                continue
            strength_model = _number_or_none(getattr(item, "strength", 1.0)) or 1.0
            append_selection(name, float(strength_model), 0.0)

    if request.preset.lower() == "ltx":
        append_distilled_loras()
        append_user_loras()
    elif request.preset.lower() == "qwen" and request.activity == "img2img":
        append_distilled_loras()
        append_user_loras()
    else:
        append_user_loras()
        append_distilled_loras()
    video_options = request.video or {}
    omnicine_enabled = video_options.get("omnicine_enabled", False)
    if isinstance(omnicine_enabled, str):
        omnicine_enabled = omnicine_enabled.lower() not in {"false", "0", "off", "none", "no"}
    if request.preset.lower() == "ltx" and omnicine_enabled is not False:
        raw_name = video_options.get("omnicine_lora") or LTX_OMNICINE_LORA_NAME
        if _is_omnicine_lora(raw_name):
            append_selection(str(raw_name), LTX_OMNICINE_DEFAULT_STRENGTH, 0.0)
    transition_enabled = video_options.get("transition_lora_enabled")
    if isinstance(transition_enabled, str):
        transition_enabled = transition_enabled.lower() not in {"false", "0", "off", "none", "no"}
    has_end_frame = bool(
        getattr(request.img2img, "reference_images", None)
        and len([item for item in request.img2img.reference_images if str(item or "").strip()]) >= 2
    )
    transition_auto = transition_enabled is True or (transition_enabled is None and has_end_frame)
    if request.preset.lower() == "ltx" and transition_auto:
        raw_name = str(video_options.get("transition_lora") or "").strip()
        if raw_name.lower() in {"", "automatic", "auto"}:
            raw_name = LTX_TRANSITION_LORA_NAME
        if raw_name.lower() not in {"none", "off", "disabled"}:
            strength = _number_or_none(video_options.get("transition_lora_strength")) or LTX_TRANSITION_DEFAULT_STRENGTH
            append_selection(raw_name, float(strength), 0.0)
    return selections


def _is_incompatible_qwen_edit_lora(name: str) -> bool:
    lower = str(name or "").lower()
    if "lightning" in lower and "qwen" not in lower:
        return True
    if "2512" in lower and "edit" not in lower:
        return True
    return False


def _flux_family_from_name(value: Any) -> str:
    lower = str(value or "").lower()
    if any(token in lower for token in ("flux-2", "flux2", "flux_2", "flux.2", "klein")):
        if "klein" in lower:
            if "9b" in lower:
                return "flux2_klein_9b"
            return "flux2_klein_4b"
        return "flux2_dev"
    return "flux1"


def _flux_lora_is_compatible(name: str, flux_family: str) -> bool:
    lower = str(name or "").replace("\\", "/").lower()
    is_flux2_model = str(flux_family or "").startswith("flux2")
    lora_is_flux2 = any(token in lower for token in ("flux2", "flux-2", "flux_2", "flux.2", "klein"))
    lora_is_flux1 = any(token in lower for token in ("flux1", "flux.1", "flux-1", "schnell", "krea", "kontext", "fill"))
    if is_flux2_model:
        if lora_is_flux1 and not lora_is_flux2:
            return False
        if "klein" in flux_family:
            return "klein" in lower
        if "klein" in lower:
            return False
        return lora_is_flux2
    return not lora_is_flux2


def _normalize_lora_name(value: Any) -> str:
    name = str(value or "").strip().replace("/", "\\")
    if not name or name.lower() in {"none", "automatic", "auto"}:
        return ""
    lower = name.lower()
    for prefix in ("loras\\", "models\\loras\\", ".\\models\\loras\\"):
        if lower.startswith(prefix):
            name = name[len(prefix) :]
            break
    if name.lower().startswith("ltx2\\"):
        name = "ltx\\" + name.split("\\", 1)[1]
    if name.lower().startswith("qwenqwen"):
        name = "qwen\\" + name[4:]
    return name


def _lora_is_compatible_with_preset(name: str, preset: str) -> bool:
    parts = [part.lower() for part in name.replace("/", "\\").split("\\") if part]
    if len(parts) < 2:
        return True
    folder = parts[0]
    known = {
        "sd15": {"sd", "sd15", "sd1", "sd1.5", "stable-diffusion"},
        "xl": {"xl", "sdxl", "illustrious", "ilustrous", "wai", "pony"},
        "qwen": {"qwen"},
        "anima": {"anima"},
        "ltx": {"ltx", "ltx2", "ltx23", "ltxv", "ltx_transition"},
        "wan": {"wan"},
        "flux": {"flux"},
        "lumina": {"lumina"},
        "zimage": {"zimage", "zimageturbo", "z-image", "z_image"},
    }
    preset_key = str(preset or "").lower()
    preset_key = {"sd": "sd15", "sd 1.5": "sd15", "sdxl": "xl", "zimageturbo": "zimage", "z-image": "zimage", "z_image": "zimage"}.get(preset_key, preset_key)
    all_known = set().union(*known.values())
    if folder not in all_known:
        return True
    return folder in known.get(preset_key, {folder})


def _effective_ltx_lora_strength(checkpoint_name: str, lora_name: str, requested_strength: float) -> float:
    strength = max(-2.0, min(2.0, float(requested_strength)))
    lower = lora_name.lower()
    if "distill" not in lower and "distilled" not in lower:
        return strength

    checkpoint_lower = str(checkpoint_name or "").lower()
    if "10eros" in checkpoint_lower:
        recommended = 0.45 if "condsafe" in lower else 0.35
    else:
        recommended = LTX_DISTILLED_CONDSAFE_DEFAULT_STRENGTH if "condsafe" in lower else LTX_DISTILLED_384_DEFAULT_STRENGTH
    if strength >= 0.95:
        return recommended
    return min(strength, recommended)


def _ltx_request_has_distilled_lora(request: GenerateRequest) -> bool:
    for item in request.distilled_loras:
        name = _normalize_lora_name(getattr(item, "name", ""))
        if name and any(token in name.lower() for token in ("distill", "distilled", "lightning", "turbo")):
            return True
    return False


def _ltx_min_steps_for_request(checkpoint_name: str, request: GenerateRequest) -> int:
    checkpoint_lower = str(checkpoint_name or "").lower()
    if any(token in checkpoint_lower for token in ("daiswa", "lightspeed", "lightning", "turbo", "distill", "distilled")):
        return 4
    if _ltx_request_has_distilled_lora(request):
        return 4
    return 8


def _ltx_latent_upscale_factor(model_name: str | None) -> float:
    text = str(model_name or "")
    match = re.search(r"x\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if match:
        try:
            factor = float(match.group(1))
        except ValueError:
            factor = 1.0
        if factor > 1.0:
            return factor
    return 2.0 if re.search(r"ltx|spatial|upscal", text, flags=re.IGNORECASE) else 1.0


def _ltx_base_dimension_for_upscale(final_dimension: int, latent_upscale_name: str | None) -> int:
    factor = _ltx_latent_upscale_factor(latent_upscale_name)
    if factor <= 1.0:
        return max(64, int(final_dimension))
    units = max(2, int(round(int(final_dimension) / factor / 32)))
    if re.search(r"ltx|spatial|upscal", str(latent_upscale_name or ""), flags=re.IGNORECASE) and units % 2:
        units += 1
    return max(64, units * 32)


def _ltx_lora_prefers_advanced_loader(lora_name: str) -> bool:
    if _is_ltx_transition_lora_name(lora_name):
        return False
    return bool(_normalize_lora_name(lora_name))


def _is_ltx_transition_lora_name(lora_name: Any) -> bool:
    text = str(lora_name or "").lower()
    return "transition" in text or "zhuanchang" in text


def _is_omnicine_lora(lora_name: Any) -> bool:
    return _ltx_lora_prefers_advanced_loader(str(lora_name or ""))


def _ltx_transition_lora_enabled(request: GenerateRequest, has_end_reference: bool = False) -> bool:
    if request.preset.lower() != "ltx":
        return False
    video_options = request.video or {}
    enabled = video_options.get("transition_lora_enabled")
    if isinstance(enabled, str):
        enabled = enabled.lower() not in {"false", "0", "off", "none", "no"}
    return enabled is True or (enabled is None and has_end_reference)


def _ltx_transition_prompt(prompt: str, request: GenerateRequest, has_end_reference: bool = False) -> str:
    if not _ltx_transition_lora_enabled(request, has_end_reference):
        return prompt
    text = str(prompt or "").strip()
    lower = text.lower()
    additions: list[str] = []
    video_options = request.video or {}
    loop_cycle = _truthy_option(video_options.get("ltx_loop_cycle"))
    if loop_cycle and "loop" not in lower:
        additions.append("seamless perfect loop cycle, first frame and last frame match naturally, no visible jump cut")
    if "transition" not in lower and LTX_TRANSITION_TRIGGER not in lower:
        additions.append(LTX_TRANSITION_PROMPT_HINT)
    if LTX_TRANSITION_TRIGGER not in lower:
        additions.append(LTX_TRANSITION_TRIGGER)
    if not additions:
        return text
    return f"{text}, {', '.join(additions)}" if text else ", ".join(additions)


def build_basic_ltx_img2video_workflow(
    request: GenerateRequest,
    checkpoint_name: str,
    text_encoder_name: str,
    reference_image_name: str,
    reference_end_image_name: str | None = None,
    base_video_name: str | None = None,
    ic_lora_name: str | None = None,
    text_projection_name: str | None = None,
    audio_vae_name: str | None = None,
    video_vae_name: str | None = None,
    latent_upscale_name: str | None = None,
    transition_lora_name: str | None = None,
    detailer_lora_name: str | None = None,
    frame_guides: list[dict[str, Any]] | None = None,
    video_combine_node: str | None = None,
    available_nodes: set[str] | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    video_options = request.video or {}
    loop_cycle = _truthy_option(video_options.get("ltx_loop_cycle"))
    loop_uses_start_as_end = loop_cycle and str(video_options.get("ltx_loop_source") or "").strip().lower() == "start_frame_as_end_frame"
    available_nodes = available_nodes or set()
    audio_volume_normalization_available = "AudioVolumeNormalization" in available_nodes
    active_audio = video_options.get("active_audio", False)
    if isinstance(active_audio, str):
        active_audio = active_audio.lower() not in {"false", "0", "off", "none", "no"}
    active_audio = bool(active_audio and audio_vae_name)
    has_end_reference = bool((reference_end_image_name or "").strip())
    frame_guides = [guide for guide in (frame_guides or []) if str(guide.get("image") or "").strip()]
    if transition_lora_name and str(video_options.get("transition_lora") or "").strip().lower() in {"", "automatic", "auto"}:
        video_options = dict(video_options)
        video_options["transition_lora"] = transition_lora_name
        request.video = video_options
    fps = max(1, int(_number_or_none(video_options.get("fps")) or 8))
    seconds = max(0.25, float(_number_or_none(video_options.get("seconds") or video_options.get("duration")) or 4.0))
    requested_frames = _number_or_none(video_options.get("frames") or video_options.get("length"))
    force_timeline_frames = bool(has_end_reference or frame_guides)
    raw_frames = int(round(requested_frames)) if requested_frames is not None and not force_timeline_frames else int(round(seconds * fps)) + 1
    length = max(9, raw_frames)
    if (length - 1) % 8 != 0:
        length = (((length - 1) // 8) + 1) * 8 + 1
    final_width = max(64, int(request.width))
    final_height = max(64, int(request.height))
    final_width -= final_width % 32
    final_height -= final_height % 32
    sampler = normalize_sampler(request.sampler or "euler_cfg_pp")
    if sampler == "euler_ancestral":
        sampler = "euler_ancestral_cfg_pp"
    min_steps = _ltx_min_steps_for_request(checkpoint_name, request)
    steps = max(min_steps, int(request.steps or min_steps))
    img_compression = max(0, min(100, int(_number_or_none(video_options.get("img_compression")) or (35 if loop_uses_start_as_end else 18))))
    use_latent_upscale = bool(latent_upscale_name)
    refine_latent_upscale_value = video_options.get("latent_upscale_refine", True)
    if isinstance(refine_latent_upscale_value, str):
        refine_latent_upscale = refine_latent_upscale_value.lower() not in {"false", "0", "off", "none", "no"}
    else:
        refine_latent_upscale = bool(refine_latent_upscale_value)
    width = final_width
    height = final_height
    if use_latent_upscale:
        width = _ltx_base_dimension_for_upscale(final_width, latent_upscale_name)
        height = _ltx_base_dimension_for_upscale(final_height, latent_upscale_name)
    ltx_model_ref: list[Any] = ["1", 0]
    ltx_lora_nodes: dict[str, Any] = {}
    next_lora_id = 40

    def add_detailer_lora(model_ref: list[Any], title: str) -> list[Any]:
        nonlocal next_lora_id
        detailer_enabled = _bool_option(video_options.get("detailer_enabled"), False)
        detailer_name = _selected_text(video_options.get("detailer_lora")) or _selected_text(detailer_lora_name)
        if not detailer_enabled or not detailer_name:
            return model_ref
        node_id = str(next_lora_id)
        strength = _number_or_none(video_options.get("detailer_strength"))
        ltx_lora_nodes[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": model_ref,
                "lora_name": detailer_name,
                "strength_model": 1.0 if strength is None else max(-2.0, min(2.0, float(strength))),
            },
            "_meta": {"title": f"{title} - {Path(detailer_name).name}"},
        }
        next_lora_id += 1
        return [node_id, 0]

    for lora_name, strength_model, _strength_clip in _active_lora_selections(request):
        if not lora_name.lower().startswith(("ltx", "ltx2", "ltx23", "ltxv")):
            continue
        node_id = str(next_lora_id)
        safe_strength = _effective_ltx_lora_strength(checkpoint_name, lora_name, strength_model)
        if _ltx_lora_prefers_advanced_loader(lora_name):
            ltx_lora_nodes[node_id] = {
                "class_type": "LTX2LoraLoaderAdvanced",
                "inputs": {
                    "lora_name": lora_name,
                    "model": ltx_model_ref,
                    "strength_model": safe_strength,
                    "video": max(0.0, min(1.0, safe_strength)),
                    "video_to_audio": max(0.0, min(1.0, safe_strength)) if active_audio else 0.0,
                    "audio": max(0.0, min(1.0, safe_strength)) if active_audio else 0.0,
                    "audio_to_video": max(0.0, min(1.0, safe_strength)) if active_audio else 0.0,
                    "other": max(0.0, min(1.0, safe_strength)),
                },
                "_meta": {"title": f"LTX LoRA - {Path(lora_name).name}"},
            }
        else:
            ltx_lora_nodes[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": ltx_model_ref,
                    "lora_name": lora_name,
                    "strength_model": safe_strength,
                },
                "_meta": {"title": f"LTX LoRA - {Path(lora_name).name}"},
            }
        ltx_model_ref = [node_id, 0]
        next_lora_id += 1

    text_to_video = request.activity == "txt2img" and not (reference_image_name or "").strip()
    motion_transfer_route = bool(
        (base_video_name or "").strip()
        and _bool_option(video_options.get("motion_transfer_enabled"), False)
        and not loop_cycle
        and not text_to_video
    )
    motion_control_mode = str(video_options.get("motion_transfer_control_mode") or "pose").strip().lower()
    if motion_control_mode not in {"pose", "canny", "depth", "camera", "raw"}:
        motion_control_mode = "pose"
    video_vae_ref: list[Any] = ["21", 0] if video_vae_name else ["1", 2]
    motion_scaffold_route = bool((base_video_name or "").strip() and video_options.get("ltx_motion_scaffold"))
    ltx_native_loop_sampler_route = bool(
        loop_uses_start_as_end
        and _bool_option(video_options.get("ltx_loop_native_sampler"), False)
        and not (base_video_name or "").strip()
        and not active_audio
        and "LTXVLoopingSampler" in available_nodes
        and "STGGuiderAdvanced" in available_nodes
    )
    loop_sampler_name = "euler" if ltx_native_loop_sampler_route and sampler in {"euler_cfg_pp", "euler_ancestral_cfg_pp"} else sampler
    loop_cfg_value = max(0.0, float(_number_or_none(request.cfg) if _number_or_none(request.cfg) is not None else 1.0))
    loop_cfg_values = ", ".join(f"{loop_cfg_value:g}" for _ in range(6))
    use_first_last_guides = bool(not text_to_video and (has_end_reference or len(frame_guides) > 1) and not (base_video_name or "").strip())
    use_ltx_flfv_reference_guides = bool(loop_uses_start_as_end and use_first_last_guides and "LTXVAddGuide" in available_nodes)
    motion_camera_end_frame_route = bool(motion_transfer_route and motion_control_mode == "camera" and has_end_reference)
    motion_transition_route = bool(motion_transfer_route and has_end_reference and _ltx_transition_lora_enabled(request, True))
    transition_route = bool(
        (use_first_last_guides or motion_scaffold_route or motion_transition_route)
        and _ltx_transition_lora_enabled(request, has_end_reference or len(frame_guides) > 1)
    )

    ic_lora_loader_ref: list[Any] | None = None
    ic_lora_downscale_ref: list[Any] | float = 1.0
    if ((base_video_name or "").strip() or transition_route) and (ic_lora_name or "").strip():
        node_id = str(next_lora_id)
        ltx_lora_nodes[node_id] = {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": ltx_model_ref,
                "lora_name": ic_lora_name,
                "strength_model": max(0.0, min(1.25, float((request.video or {}).get("ltx_ic_lora_strength") or (0.65 if transition_route else 1.0)))),
            },
            "_meta": {"title": f"LTX IC-LoRA Motion - {Path(str(ic_lora_name)).name}"},
        }
        ic_lora_loader_ref = [node_id, 0]
        ic_lora_downscale_ref = [node_id, 1]
        ltx_model_ref = ic_lora_loader_ref
        next_lora_id += 1

    positive_prompt_text = request.prompt
    if request.preset.lower() == "ltx":
        if (
            not str(positive_prompt_text or "").strip()
            and not text_to_video
            and (reference_image_name or "").strip()
            and not has_end_reference
            and not frame_guides
            and not (base_video_name or "").strip()
            and not loop_cycle
        ):
            positive_prompt_text = (
                "preserve the same subject, identity, pose, clothing, room and lighting from the reference image; "
                "natural subtle video motion, clean recognizable details"
            )
        active_lora_names = [name.lower() for name, _strength, _clip_strength in _active_lora_selections(request)]
        if any("livewallpaper_ltx23_r64_6250" in name for name in active_lora_names) and "l1v3w4llp4p3r" not in str(positive_prompt_text).lower():
            positive_prompt_text = f"{positive_prompt_text}, l1v3w4llp4p3r" if str(positive_prompt_text or "").strip() else "l1v3w4llp4p3r"
    positive_prompt = _ltx_transition_prompt(positive_prompt_text, request, has_end_reference)
    use_ltx_nag_model = bool(transition_route or (motion_transfer_route and motion_control_mode == "camera"))
    transition_model_ref: list[Any] = ltx_model_ref
    if use_ltx_nag_model:
        transition_model_ref = ["37", 0]
    main_model_ref: list[Any] = transition_model_ref
    if not (use_latent_upscale and refine_latent_upscale):
        main_model_ref = add_detailer_lora(main_model_ref, "LTX Detailer LoRA")
    sampler_latent_ref: list[Any] = ["7", 0] if (text_to_video or motion_transfer_route or ltx_native_loop_sampler_route) else ["7", 2]
    scheduler_latent_ref: list[Any] = sampler_latent_ref
    ltx_positive_ref: list[Any] = ["6", 0] if (text_to_video or motion_transfer_route or ltx_native_loop_sampler_route) else ["7", 0]
    ltx_negative_ref: list[Any] = ["6", 1] if (text_to_video or motion_transfer_route or ltx_native_loop_sampler_route) else ["7", 1]
    if use_first_last_guides:
        sampler_latent_ref = ["7", 0]
        scheduler_latent_ref = ["7", 0]
        ltx_positive_ref = ["6", 0]
        ltx_negative_ref = ["6", 1]
    decode_latent_ref: list[Any] = ["12", 0]
    create_video_image_ref: list[Any] = ["13", 0]
    create_video_inputs: dict[str, Any] = {"images": create_video_image_ref, "fps": float(fps)}
    has_audio_context = bool(audio_vae_name and (active_audio or motion_transfer_route))
    first_audio_latent_ref: list[Any] | None = None

    preprocess_nodes: dict[str, Any] = {}
    motion_guides_cropped_before_sampling = False
    use_ic_timeline_guides = _bool_option(video_options.get("ltx_ic_timeline_guides"), False)
    reference_image_ref: list[Any] = ["3", 0]
    if not text_to_video:
        start_frame_image_ref: list[Any] = ["3", 0]
        resize_start_reference = not motion_transfer_route
        if resize_start_reference:
            preprocess_nodes["171"] = {
                "class_type": "ImageResizeKJv2",
                "inputs": {
                    "image": ["3", 0],
                    "width": width,
                    "height": height,
                    "upscale_method": "lanczos",
                    "keep_proportion": "resize",
                    "pad_color": "0, 0, 0",
                    "crop_position": "center",
                    "divisible_by": 32,
                    "device": "cpu",
                },
                "_meta": {"title": "Resize LTX Start Frame To Requested Size"},
            }
            start_frame_image_ref = ["171", 0]
        preprocess_nodes["22"] = {
            "class_type": "LTXVPreprocess",
            "inputs": {"image": start_frame_image_ref, "img_compression": img_compression},
            "_meta": {"title": "LTX Start Frame Preprocess"},
        }
        reference_image_ref = ["22", 0]
        has_end_frame = has_end_reference
        use_motion_scaffold = bool((base_video_name or "").strip() and video_options.get("ltx_motion_scaffold"))
        guide_specs: list[dict[str, Any]] = []
        if (has_end_frame or len(frame_guides) > 1) and not use_motion_scaffold and not motion_camera_end_frame_route:
            if frame_guides:
                for guide in frame_guides[:8]:
                    guide_index_value = int(_number_or_none(guide.get("index")) if _number_or_none(guide.get("index")) is not None else 0)
                    if guide_index_value < 0:
                        guide_index_value = max(0, length + guide_index_value)
                    guide_specs.append(
                        {
                            "image": str(guide.get("image") or "").strip(),
                            "index": guide_index_value,
                            "strength": max(0.0, min(1.0, float(_number_or_none(guide.get("strength")) or 1.0))),
                        }
                    )
            else:
                start_strength_value = _number_or_none(video_options.get("start_frame_strength"))
                end_strength_value = _number_or_none(video_options.get("end_frame_strength"))
                guide_specs = [
                    {
                        "image": reference_image_name,
                        "index": 0,
                            "strength": max(0.0, min(1.0, float(
                            start_strength_value if start_strength_value is not None else (0.70 if loop_uses_start_as_end else 1.0)
                        ))),
                    },
                    {
                        "image": str(reference_end_image_name or ""),
                        "index": length - 1,
                            "strength": max(0.0, min(1.0, float(
                            end_strength_value if end_strength_value is not None else (0.70 if loop_uses_start_as_end else 1.0)
                        ))),
                    },
                ]
            use_ic_timeline_guides = _bool_option(video_options.get("ltx_ic_timeline_guides"), False)
            in_place_inputs: dict[str, Any] = {"vae": video_vae_ref, "latent": sampler_latent_ref, "num_images": str(len(guide_specs))}
            for guide_index, guide in enumerate(guide_specs, start=1):
                load_id = "3" if guide_index == 1 and guide["image"] == reference_image_name else str(180 + guide_index)
                resize_id = str(200 + guide_index)
                preprocess_id = str(190 + guide_index)
                if load_id != "3":
                    preprocess_nodes[load_id] = {
                        "class_type": "LoadImage",
                        "inputs": {"image": guide["image"]},
                        "_meta": {"title": f"LTX Timeline Frame {guide_index}"},
                    }
                preprocess_nodes[resize_id] = {
                    "class_type": "ImageResizeKJv2",
                    "inputs": {
                        "image": ["171", 0] if load_id == "3" else [load_id, 0],
                        "width": width,
                        "height": height,
                        "upscale_method": "lanczos",
                        "keep_proportion": "resize",
                        "pad_color": "0, 0, 0",
                        "crop_position": "center",
                        "divisible_by": 32,
                        "device": "cpu",
                    },
                    "_meta": {"title": f"Resize LTX Timeline Frame {guide_index}"},
                }
                preprocess_nodes[preprocess_id] = {
                    "class_type": "LTXVPreprocess",
                    "inputs": {"image": [resize_id, 0], "img_compression": img_compression},
                    "_meta": {"title": f"LTX Timeline Frame {guide_index} Preprocess"},
                }
                in_place_inputs[f"num_images.image_{guide_index}"] = [preprocess_id, 0]
                in_place_inputs[f"num_images.strength_{guide_index}"] = guide["strength"]
                in_place_inputs[f"num_images.index_{guide_index}"] = guide["index"]
            if use_ic_timeline_guides and transition_route and ic_lora_loader_ref:
                for guide_index, guide in enumerate(guide_specs, start=1):
                    guide_node_id = str(75 + guide_index)
                    resize_id = str(200 + guide_index)
                    preprocess_nodes[guide_node_id] = {
                        "class_type": "LTXAddVideoICLoRAGuide",
                        "inputs": {
                            "positive": ltx_positive_ref,
                            "negative": ltx_negative_ref,
                            "vae": video_vae_ref,
                            "latent": sampler_latent_ref,
                            "image": [resize_id, 0],
                            "frame_idx": int(guide["index"]),
                            "strength": max(0.0, min(1.0, float(_number_or_none(guide.get("strength")) or 1.0))),
                            "latent_downscale_factor": ic_lora_downscale_ref,
                            "crop": str(video_options.get("ltx_ic_crop") or "disabled"),
                            "use_tiled_encode": bool(video_options.get("ltx_ic_tiled_encode") or False),
                            "tile_size": int(_number_or_none(video_options.get("ltx_ic_tile_size")) or 256),
                            "tile_overlap": int(_number_or_none(video_options.get("ltx_ic_tile_overlap")) or 64),
                        },
                        "_meta": {"title": f"LTX IC-LoRA Timeline Guide {guide_index}"},
                    }
                    ltx_positive_ref = [guide_node_id, 0]
                    ltx_negative_ref = [guide_node_id, 1]
                    sampler_latent_ref = [guide_node_id, 2]
                    scheduler_latent_ref = [guide_node_id, 2]
            elif use_ltx_flfv_reference_guides:
                first_guide = guide_specs[0]
                last_guide = guide_specs[-1]
                preprocess_nodes["76"] = {
                    "class_type": "LTXVAddGuide",
                    "inputs": {
                        "positive": ltx_positive_ref,
                        "negative": ltx_negative_ref,
                        "vae": video_vae_ref,
                        "latent": sampler_latent_ref,
                        "image": reference_image_ref,
                        "frame_idx": 0,
                        "strength": max(0.0, min(1.0, float(_number_or_none(first_guide.get("strength")) or 0.7))),
                    },
                    "_meta": {"title": "LTX FLF2V Guide: Start Frame"},
                }
                last_positive_ref = ["76", 0]
                last_negative_ref = ["76", 1]
                last_latent_ref = ["76", 2]
                preprocess_nodes["178"] = {
                    "class_type": "LTXVAddGuide",
                    "inputs": {
                        "positive": last_positive_ref,
                        "negative": last_negative_ref,
                        "vae": video_vae_ref,
                        "latent": last_latent_ref,
                        "image": reference_image_ref,
                        "frame_idx": -1,
                        "strength": max(0.0, min(1.0, float(_number_or_none(last_guide.get("strength")) or 0.7))),
                    },
                    "_meta": {"title": "LTX FLF2V Guide: End Frame"},
                }
                ltx_positive_ref = ["178", 0]
                ltx_negative_ref = ["178", 1]
                sampler_latent_ref = ["178", 2]
                scheduler_latent_ref = ["178", 2]
            else:
                preprocess_nodes["76"] = {
                    "class_type": "LTXVImgToVideoInplaceKJ",
                    "inputs": in_place_inputs,
                    "_meta": {"title": "LTX Official Inplace Timeline Frames"},
                }
                sampler_latent_ref = ["76", 0]
                scheduler_latent_ref = ["76", 0]
        if motion_camera_end_frame_route and (reference_end_image_name or "").strip() and not guide_specs:
            guide_specs = [
                {
                    "image": reference_image_name,
                    "index": 0,
                    "strength": max(0.0, min(1.0, float(_number_or_none(video_options.get("motion_transfer_target_strength")) or 0.2))),
                },
                {
                    "image": str(reference_end_image_name or ""),
                    "index": length - 1,
                    "strength": max(0.0, min(1.0, float(_number_or_none(video_options.get("end_frame_strength")) or 0.7))),
                },
            ]
        if (base_video_name or "").strip():
            motion_transfer_enabled = motion_transfer_route
            motion_strength = float(
                _number_or_none(video_options.get("motion_strength") or video_options.get("video_strength"))
                or request.img2img.denoise
                or 0.95
            )
            if motion_transfer_enabled:
                motion_strength = float(_number_or_none(video_options.get("motion_transfer_motion_strength")) or 1.0)
            if motion_scaffold_route:
                motion_control_mode = str(video_options.get("ltx_start_end_control_mode") or "canny").strip().lower()
            use_camera_reference_frames = bool(
                motion_transfer_enabled
                and motion_control_mode == "camera"
                and "VHS_LoadVideo" in available_nodes
            )
            preprocess_nodes["72"] = {
                "class_type": "VHS_LoadVideo" if use_camera_reference_frames else ("LoadVideo" if (motion_transfer_enabled or motion_scaffold_route) else "VHS_LoadVideo"),
                "inputs": (
                    {
                        "video": base_video_name,
                        "force_rate": float(fps),
                        "custom_width": 0,
                        "custom_height": 0,
                        "frame_load_cap": length,
                        "skip_first_frames": 0,
                        "select_every_nth": 1,
                        "format": "LTXV",
                    }
                    if use_camera_reference_frames
                    else (
                    {"file": base_video_name}
                    if (motion_transfer_enabled or motion_scaffold_route)
                    else {
                        "video": base_video_name,
                        "force_rate": float(fps),
                        "custom_width": width,
                        "custom_height": height,
                        "frame_load_cap": length,
                        "skip_first_frames": 0,
                        "select_every_nth": 1,
                        "format": "LTXV",
                    }
                    )
                ),
                "_meta": {"title": "LTX Cameraman Reference Video" if use_camera_reference_frames else "LTX Base Motion Video"},
            }
            if motion_transfer_enabled or motion_scaffold_route:
                motion_resize_source_ref: list[Any] = ["72", 0]
                if not use_camera_reference_frames:
                    preprocess_nodes["74"] = {
                        "class_type": "GetVideoComponents",
                        "inputs": {"video": ["72", 0]},
                        "_meta": {"title": "LTX Motion Video Components"},
                    }
                    motion_resize_source_ref = ["74", 0]
                preprocess_nodes["75"] = {
                    "class_type": "ImageResizeKJv2",
                    "inputs": {
                        "image": motion_resize_source_ref,
                        "width": width,
                        "height": height,
                        "upscale_method": "lanczos",
                        "keep_proportion": "crop",
                        "pad_color": "0, 0, 0",
                        "crop_position": "center",
                        "divisible_by": 32,
                        "device": "cpu",
                    },
                    "_meta": {"title": "Resize LTX Motion Frames Like Official Workflow" if motion_transfer_enabled else "Resize LTX Start/End Scaffold Frames"},
                }
            if motion_transfer_enabled:
                target_condition_image_ref: list[Any] = start_frame_image_ref
                if motion_control_mode != "camera" and "ResizeImageMaskNode" in available_nodes:
                    preprocess_nodes["78"] = {
                        "class_type": "ResizeImageMaskNode",
                        "inputs": {
                            "input": start_frame_image_ref,
                            "resize_type": "scale longer dimension",
                            "resize_type.longer_size": max(1536, max(width, height)),
                            "scale_method": "lanczos",
                        },
                        "_meta": {"title": "Resize LTX Motion Target Like Official IC Workflow"},
                    }
                    target_condition_image_ref = ["78", 0]
                elif "ImageResizeKJv2" in available_nodes:
                    preprocess_nodes["78"] = {
                        "class_type": "ImageResizeKJv2",
                        "inputs": {
                            "image": start_frame_image_ref,
                            "width": width,
                            "height": height,
                            "upscale_method": "lanczos",
                            "keep_proportion": "crop",
                            "pad_color": "0, 0, 0",
                            "crop_position": "center",
                            "divisible_by": 32,
                            "device": "cpu",
                        },
                        "_meta": {"title": "Resize LTX Cameraman Target Like Official Workflow" if motion_control_mode == "camera" else "Resize LTX Motion Target Like IC Workflow"},
                    }
                    target_condition_image_ref = ["78", 0]
                if not motion_transition_route and not motion_camera_end_frame_route:
                    target_strength = max(0.0, min(1.0, float(_number_or_none(video_options.get("motion_transfer_target_strength")) or 1.0)))
                    if motion_control_mode == "camera":
                        preprocess_nodes["79"] = {
                            "class_type": "LTXVImgToVideoConditionOnly",
                            "inputs": {
                                "vae": video_vae_ref,
                                "image": target_condition_image_ref,
                                "latent": sampler_latent_ref,
                                "strength": target_strength,
                                "bypass": _bool_option(video_options.get("ltx_ic_image_bypass"), False),
                            },
                            "_meta": {"title": "LTX Cameraman First Frame Conditioning"},
                        }
                    else:
                        preprocess_nodes["79"] = {
                            "class_type": "LTXVImgToVideoConditionOnly",
                            "inputs": {
                                "vae": video_vae_ref,
                                "image": target_condition_image_ref,
                                "latent": sampler_latent_ref,
                                "strength": target_strength,
                                "bypass": _bool_option(video_options.get("ltx_ic_image_bypass"), False),
                            },
                            "_meta": {"title": "LTX Motion Transfer Target Condition (Official IC Bypass)"},
                        }
                    sampler_latent_ref = ["79", 0]
                    scheduler_latent_ref = ["79", 0]
            motion_guide_image_ref: list[Any] = ["72", 0]
            if motion_transfer_enabled or motion_scaffold_route:
                motion_guide_image_ref = ["75", 0]
                if motion_control_mode == "pose" and ("DWPreprocessor" in available_nodes or "OpenposePreprocessor" in available_nodes):
                    pose_node = "DWPreprocessor" if "DWPreprocessor" in available_nodes else "OpenposePreprocessor"
                    pose_inputs: dict[str, Any] = {
                        "image": ["75", 0],
                        "detect_hand": "enable",
                        "detect_body": "enable",
                        "detect_face": "enable",
                        "resolution": max(width, height),
                        "scale_stick_for_xinsr_cn": "disable",
                    }
                    if pose_node == "DWPreprocessor":
                        pose_inputs["bbox_detector"] = "yolox_l.onnx"
                        pose_inputs["pose_estimator"] = "dw-ll_ucoco_384_bs5.torchscript.pt"
                    preprocess_nodes["271"] = {
                        "class_type": pose_node,
                        "inputs": pose_inputs,
                        "_meta": {"title": "LTX Motion Transfer Pose / DWPose"},
                    }
                    motion_guide_image_ref = ["271", 0]
                elif motion_control_mode == "canny" and ("CannyEdgePreprocessor" in available_nodes or "Canny" in available_nodes):
                    canny_node = "CannyEdgePreprocessor" if "CannyEdgePreprocessor" in available_nodes else "Canny"
                    canny_inputs: dict[str, Any] = {
                        "image": ["75", 0],
                    }
                    if canny_node == "CannyEdgePreprocessor":
                        canny_inputs["low_threshold"] = int(_number_or_none(video_options.get("motion_transfer_canny_low")) or 92)
                        canny_inputs["high_threshold"] = int(_number_or_none(video_options.get("motion_transfer_canny_high")) or 200)
                        canny_inputs["resolution"] = max(width, height)
                    else:
                        canny_inputs["low_threshold"] = float(_number_or_none(video_options.get("motion_transfer_canny_low")) or 0.4)
                        canny_inputs["high_threshold"] = float(_number_or_none(video_options.get("motion_transfer_canny_high")) or 0.8)
                    preprocess_nodes["271"] = {
                        "class_type": canny_node,
                        "inputs": canny_inputs,
                        "_meta": {"title": "LTX Start/End Scaffold Canny IC Control" if motion_scaffold_route else "LTX Motion Transfer Canny"},
                    }
                    motion_guide_image_ref = ["271", 0]
                elif motion_control_mode == "camera":
                    # Cameraman IC-LoRA uses the resized reference video frames directly, unlike Pose/Canny/Depth.
                    motion_guide_image_ref = ["75", 0]
                elif motion_control_mode == "depth" and {"LoadVideoDepthAnythingModel", "VideoDepthAnythingProcess", "VideoDepthAnythingOutput"}.issubset(available_nodes):
                    preprocess_nodes["271"] = {
                        "class_type": "LoadVideoDepthAnythingModel",
                        "inputs": {"model": str(video_options.get("motion_transfer_depth_model") or "video_depth_anything_vits.pth")},
                        "_meta": {"title": "Load LTX Motion Transfer Depth Model"},
                    }
                    preprocess_nodes["272"] = {
                        "class_type": "VideoDepthAnythingProcess",
                        "inputs": {
                            "vda_model": ["271", 0],
                            "images": ["75", 0],
                            "input_size": int(_number_or_none(video_options.get("motion_transfer_depth_input_size")) or 518),
                            "max_res": max(width, height),
                            "precision": "fp32",
                        },
                        "_meta": {"title": "LTX Motion Transfer Depth"},
                    }
                    preprocess_nodes["273"] = {
                        "class_type": "VideoDepthAnythingOutput",
                        "inputs": {"depths": ["272", 0], "colormap": "gray"},
                        "_meta": {"title": "LTX Motion Transfer Depth Images"},
                    }
                    motion_guide_image_ref = ["273", 0]
                elif motion_scaffold_route and ("CannyEdgePreprocessor" in available_nodes or "Canny" in available_nodes):
                    canny_node = "CannyEdgePreprocessor" if "CannyEdgePreprocessor" in available_nodes else "Canny"
                    canny_inputs: dict[str, Any] = {"image": ["72", 0]}
                    if canny_node == "CannyEdgePreprocessor":
                        canny_inputs["low_threshold"] = int(_number_or_none(video_options.get("ltx_start_end_canny_low")) or 80)
                        canny_inputs["high_threshold"] = int(_number_or_none(video_options.get("ltx_start_end_canny_high")) or 180)
                        canny_inputs["resolution"] = max(width, height)
                    else:
                        canny_inputs["low_threshold"] = float(_number_or_none(video_options.get("ltx_start_end_canny_low")) or 0.35)
                        canny_inputs["high_threshold"] = float(_number_or_none(video_options.get("ltx_start_end_canny_high")) or 0.75)
                    preprocess_nodes["274"] = {
                        "class_type": canny_node,
                        "inputs": canny_inputs,
                        "_meta": {"title": "LTX Start/End Scaffold Canny IC Control"},
                    }
                    motion_guide_image_ref = ["274", 0]
            if (motion_transfer_enabled or motion_scaffold_route) and motion_control_mode != "camera" and "ResizeImageMaskNode" in available_nodes:
                preprocess_nodes["84"] = {
                    "class_type": "SimpleMath+",
                    "inputs": {
                        "a": ic_lora_downscale_ref,
                        "value": "a*32",
                    },
                    "_meta": {"title": "LTX IC-LoRA Resize Multiple"},
                }
                preprocess_nodes["83"] = {
                    "class_type": "ResizeImageMaskNode",
                    "inputs": {
                        "input": motion_guide_image_ref,
                        "resize_type": "scale to multiple",
                        "resize_type.multiple": ["84", 0],
                        "scale_method": "lanczos",
                    },
                    "_meta": {"title": "Resize LTX Motion Control Like Official IC Workflow"},
                }
                motion_guide_image_ref = ["83", 0]
            if ic_lora_loader_ref:
                if motion_transfer_enabled:
                    preprocess_nodes["82"] = {
                        "class_type": "GetImageSize",
                        "inputs": {"image": motion_guide_image_ref},
                        "_meta": {"title": "Measure LTX Motion Guide Frames"},
                    }
                    preprocess_nodes["7"] = {
                        "class_type": "EmptyLTXVLatentVideo",
                        "inputs": {
                            "width": ["82", 0],
                            "height": ["82", 1],
                            "length": ["82", 2],
                            "batch_size": max(1, request.batch_size),
                        },
                        "_meta": {"title": "LTX Motion Transfer Latent From Guide Frames"},
                    }
                    sampler_latent_ref = ["7", 0]
                    scheduler_latent_ref = ["7", 0]
                    if (motion_transition_route or motion_camera_end_frame_route) and (reference_end_image_name or "").strip():
                        preprocess_nodes["86"] = {
                            "class_type": "LoadImage",
                            "inputs": {"image": str(reference_end_image_name)},
                            "_meta": {"title": "LTX Motion Transfer Optional End Frame"},
                        }
                        preprocess_nodes["87"] = {
                            "class_type": "ImageResizeKJv2",
                            "inputs": {
                                "image": ["86", 0],
                                "width": width,
                                "height": height,
                                "upscale_method": "lanczos",
                                "keep_proportion": "crop",
                                "pad_color": "0, 0, 0",
                                "crop_position": "center",
                                "divisible_by": 32,
                                "device": "cpu",
                            },
                            "_meta": {"title": "Resize LTX Motion Transfer End Frame"},
                        }
                        guide_strength_default = (
                            max(0.0, min(1.0, float(_number_or_none(video_options.get("motion_transfer_target_strength")) or 0.2)))
                            if motion_control_mode == "camera"
                            else 1.0
                        )
                        start_strength = max(0.0, min(1.0, float(_number_or_none(video_options.get("start_frame_strength")) or guide_strength_default)))
                        end_strength = max(0.0, min(1.0, float(_number_or_none(video_options.get("end_frame_strength")) or guide_strength_default)))
                        preprocess_nodes["89"] = {
                            "class_type": "LTXVImgToVideoInplaceKJ",
                            "inputs": {
                                "vae": video_vae_ref,
                                "latent": sampler_latent_ref,
                                "num_images": "2",
                                "num_images.image_1": target_condition_image_ref,
                                "num_images.strength_1": start_strength,
                                "num_images.index_1": 0,
                                "num_images.image_2": ["87", 0],
                                "num_images.strength_2": end_strength,
                                "num_images.index_2": -1,
                            },
                            "_meta": {"title": "LTX Motion Transfer Target + End Frame Before Camera Guide"},
                        }
                        sampler_latent_ref = ["89", 0]
                        scheduler_latent_ref = ["89", 0]
                    elif "79" in preprocess_nodes:
                        preprocess_nodes["79"]["inputs"]["latent"] = sampler_latent_ref
                        sampler_latent_ref = ["79", 0]
                        scheduler_latent_ref = ["79", 0]
                ic_guide_crop = str(video_options.get("ltx_ic_crop") or "disabled")
                ic_guide_tiled_encode = bool(video_options.get("ltx_ic_tiled_encode") or False)
                preprocess_nodes["73"] = {
                    "class_type": "LTXAddVideoICLoRAGuide",
                    "inputs": {
                        "positive": ltx_positive_ref,
                        "negative": ltx_negative_ref,
                        "vae": video_vae_ref,
                        "latent": sampler_latent_ref,
                        "image": motion_guide_image_ref,
                        "frame_idx": 0,
                        "strength": max(0.0, min(1.0, motion_strength)),
                        "latent_downscale_factor": ic_lora_downscale_ref,
                        "crop": ic_guide_crop,
                        "use_tiled_encode": ic_guide_tiled_encode,
                        "tile_size": int(_number_or_none(video_options.get("ltx_ic_tile_size")) or 256),
                        "tile_overlap": int(_number_or_none(video_options.get("ltx_ic_tile_overlap")) or 64),
                    },
                    "_meta": {"title": f"Guide LTX Motion With IC-LoRA ({motion_control_mode})"},
                }
            else:
                preprocess_nodes["73"] = {
                    "class_type": "LTXVAddGuide",
                    "inputs": {
                        "positive": ltx_positive_ref,
                        "negative": ltx_negative_ref,
                        "vae": video_vae_ref,
                        "latent": sampler_latent_ref,
                        "image": ["72", 0],
                        "frame_idx": 0,
                        "strength": max(0.0, min(1.0, motion_strength)),
                    },
                    "_meta": {"title": "Guide LTX Motion Video"},
                }
            ltx_positive_ref = ["73", 0]
            ltx_negative_ref = ["73", 1]
            sampler_latent_ref = ["73", 2]
            scheduler_latent_ref = ["73", 2]
            crop_motion_guides_before_sampling = False
            if crop_motion_guides_before_sampling and motion_transfer_enabled and motion_control_mode == "camera" and "LTXVCropGuides" in available_nodes:
                preprocess_nodes["81"] = {
                    "class_type": "LTXVCropGuides",
                    "inputs": {
                        "positive": ltx_positive_ref,
                        "negative": ltx_negative_ref,
                        "latent": sampler_latent_ref,
                    },
                    "_meta": {"title": "Crop LTX Motion Transfer Guides Before Sampling"},
                }
                ltx_positive_ref = ["81", 0]
                ltx_negative_ref = ["81", 1]
                sampler_latent_ref = ["81", 2]
                scheduler_latent_ref = ["81", 2]
                motion_guides_cropped_before_sampling = True

    audio_nodes: dict[str, Any] = {}
    if has_audio_context:
        audio_nodes = {
            "16": {
                "class_type": "VAELoader",
                "inputs": {"vae_name": audio_vae_name},
                "_meta": {"title": "Load LTX Audio VAE"},
            },
            "17": {
                "class_type": "LTXVEmptyLatentAudio",
                "inputs": {
                    "frames_number": length,
                    "frame_rate": int(fps),
                    "batch_size": max(1, request.batch_size),
                    "audio_vae": ["16", 0],
                },
                "_meta": {"title": "Empty LTX Audio Latent"},
            },
            "18": {
                "class_type": "LTXVConcatAVLatent",
                "inputs": {"video_latent": sampler_latent_ref, "audio_latent": ["17", 0]},
                "_meta": {"title": "Merge Video + Audio Latents"},
            },
            "19": {
                "class_type": "LTXVSeparateAVLatent",
                "inputs": {"av_latent": ["12", 0]},
                "_meta": {"title": "Separate Video + Audio Latents"},
            },
        }
        sampler_latent_ref = ["18", 0]
        scheduler_latent_ref = ["18", 0]
        first_audio_latent_ref = ["19", 1]
        decode_latent_ref = ["19", 0]

    video_vae_nodes: dict[str, Any] = {}
    if video_vae_name:
        video_vae_nodes["21"] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": video_vae_name},
            "_meta": {"title": "Load LTX Video VAE"},
        }

    refiner_positive_ref: list[Any] = ltx_positive_ref
    refiner_negative_ref: list[Any] = ltx_negative_ref
    latent_upscale_nodes: dict[str, Any] = {}
    if use_latent_upscale:
        crop_ic_guides_before_upscale = bool(
            motion_transfer_route or use_ltx_flfv_reference_guides or (transition_route and use_ic_timeline_guides and ic_lora_loader_ref)
        )
        if crop_ic_guides_before_upscale:
            latent_upscale_nodes["540"] = {
                "class_type": "LTXVCropGuides",
                "inputs": {
                    "positive": ltx_positive_ref,
                    "negative": ltx_negative_ref,
                    "latent": decode_latent_ref,
                },
                "_meta": {"title": "Crop LTX IC Guides Before Latent Upscale"},
            }
            refiner_positive_ref = ["540", 0]
            refiner_negative_ref = ["540", 1]
            decode_latent_ref = ["540", 2]
        latent_upscale_nodes["23"] = {
            "class_type": "LatentUpscaleModelLoader",
            "inputs": {"model_name": latent_upscale_name},
            "_meta": {"title": "Load LTX Latent Upscaler"},
        }
        latent_upscale_nodes["24"] = {
            "class_type": "LTXVLatentUpsampler",
            "inputs": {"samples": decode_latent_ref, "upscale_model": ["23", 0], "vae": video_vae_ref},
            "_meta": {"title": "LTX Latent Upscale"},
        }
        refiner_latent_ref: list[Any] = ["24", 0]
        if not text_to_video:
            if ((reference_end_image_name or "").strip() or len(frame_guides) > 1) and not use_motion_scaffold and not motion_camera_end_frame_route:
                if not crop_ic_guides_before_upscale:
                    refiner_positive_ref = ltx_positive_ref
                    refiner_negative_ref = ltx_negative_ref
                if not (use_ic_timeline_guides and transition_route and ic_lora_loader_ref):
                    use_guide_multi_after_upscale = bool((motion_transition_route or motion_camera_end_frame_route) and "LTXVAddGuideMulti" in available_nodes)
                    upscale_in_place_inputs: dict[str, Any] = {
                        "vae": video_vae_ref,
                        "latent": ["24", 0],
                        "num_images": str(len(guide_specs)),
                    }
                    upscale_guide_multi_inputs: dict[str, Any] = {
                        "positive": refiner_positive_ref,
                        "negative": refiner_negative_ref,
                        "vae": video_vae_ref,
                        "latent": ["24", 0],
                        "num_guides": str(len(guide_specs)),
                    }
                    for guide_index, guide in enumerate(guide_specs, start=1):
                        load_id = "3" if guide_index == 1 and guide["image"] == reference_image_name else str(550 + guide_index)
                        resize_id = str(560 + guide_index)
                        preprocess_id = str(570 + guide_index)
                        if load_id != "3":
                            latent_upscale_nodes[load_id] = {
                                "class_type": "LoadImage",
                                "inputs": {"image": guide["image"]},
                                "_meta": {"title": f"LTX Upscale Timeline Frame {guide_index}"},
                            }
                        latent_upscale_nodes[resize_id] = {
                            "class_type": "ImageResizeKJv2",
                            "inputs": {
                                "image": [load_id, 0],
                                "width": final_width,
                                "height": final_height,
                                "upscale_method": "lanczos",
                                "keep_proportion": "resize",
                                "pad_color": "0, 0, 0",
                                "crop_position": "center",
                                "divisible_by": 32,
                                "device": "cpu",
                            },
                            "_meta": {"title": f"Resize LTX Upscale Timeline Frame {guide_index}"},
                        }
                        if use_guide_multi_after_upscale:
                            upscale_guide_multi_inputs[f"num_guides.image_{guide_index}"] = [resize_id, 0]
                            upscale_guide_multi_inputs[f"num_guides.frame_idx_{guide_index}"] = guide["index"]
                            upscale_guide_multi_inputs[f"num_guides.strength_{guide_index}"] = guide["strength"]
                        else:
                            latent_upscale_nodes[preprocess_id] = {
                                "class_type": "LTXVPreprocess",
                                "inputs": {"image": [resize_id, 0], "img_compression": img_compression},
                                "_meta": {"title": f"LTX Upscale Timeline Frame {guide_index} Preprocess"},
                            }
                            upscale_in_place_inputs[f"num_images.image_{guide_index}"] = [preprocess_id, 0]
                            upscale_in_place_inputs[f"num_images.strength_{guide_index}"] = guide["strength"]
                            upscale_in_place_inputs[f"num_images.index_{guide_index}"] = guide["index"]
                    if use_guide_multi_after_upscale:
                        latent_upscale_nodes["25"] = {
                            "class_type": "LTXVAddGuideMulti",
                            "inputs": upscale_guide_multi_inputs,
                            "_meta": {"title": "Reapply LTX Camera Start/End Guides After Latent Upscale"},
                        }
                        refiner_positive_ref = ["25", 0]
                        refiner_negative_ref = ["25", 1]
                        refiner_latent_ref = ["25", 2]
                    else:
                        latent_upscale_nodes["25"] = {
                            "class_type": "LTXVImgToVideoInplaceKJ",
                            "inputs": upscale_in_place_inputs,
                            "_meta": {"title": "Reapply LTX Start/End Frames After Latent Upscale"},
                        }
                        refiner_latent_ref = ["25", 0]
            else:
                use_camera_inplace_after_upscale = False
                if use_camera_inplace_after_upscale and motion_transfer_route and motion_control_mode == "camera":
                    camera_upscale_strength = max(0.0, min(1.0, float(_number_or_none(video_options.get("motion_transfer_target_strength")) if _number_or_none(video_options.get("motion_transfer_target_strength")) is not None else 0.2)))
                    if "LTXVImgToVideoInplace" in available_nodes:
                        latent_upscale_nodes["25"] = {
                            "class_type": "LTXVImgToVideoInplace",
                            "inputs": {
                                "vae": video_vae_ref,
                                "image": reference_image_ref,
                                "latent": ["24", 0],
                                "strength": camera_upscale_strength,
                                "bypass": False,
                            },
                            "_meta": {"title": "Reapply LTX Camera Reference After Latent Upscale"},
                        }
                        refiner_latent_ref = ["25", 0]
                    elif "LTXVImgToVideoInplaceKJ" in available_nodes:
                        latent_upscale_nodes["25"] = {
                            "class_type": "LTXVImgToVideoInplaceKJ",
                            "inputs": {
                                "vae": video_vae_ref,
                                "latent": ["24", 0],
                                "num_images": "1",
                                "num_images.image_1": reference_image_ref,
                                "num_images.strength_1": camera_upscale_strength,
                                "num_images.index_1": 0,
                            },
                            "_meta": {"title": "Reapply LTX Camera Reference After Latent Upscale"},
                        }
                        refiner_latent_ref = ["25", 0]
                    else:
                        latent_upscale_nodes["25"] = {
                            "class_type": "LTXVImgToVideoConditionOnly",
                            "inputs": {
                                "vae": video_vae_ref,
                                "image": reference_image_ref,
                                "latent": ["24", 0],
                                "strength": camera_upscale_strength,
                            },
                            "_meta": {"title": "Reapply LTX Camera Reference After Latent Upscale"},
                        }
                        refiner_latent_ref = ["25", 0]
                else:
                    upscale_condition_strength = (
                        max(0.0, min(1.0, float(_number_or_none(video_options.get("motion_transfer_target_strength")) or 0.7)))
                        if motion_transfer_route
                        else float(video_options.get("upscale_condition_strength") or 1.0)
                    )
                    latent_upscale_nodes["25"] = {
                        "class_type": "LTXVImgToVideoConditionOnly",
                        "inputs": {
                            "vae": video_vae_ref,
                            "image": reference_image_ref,
                            "latent": ["24", 0],
                            "strength": upscale_condition_strength,
                        },
                        "_meta": {"title": "Reapply Reference After Latent Upscale"},
                    }
                    refiner_latent_ref = ["25", 0]
        if refine_latent_upscale:
            if not text_to_video and not motion_transfer_route and (not (reference_end_image_name or "").strip() or use_motion_scaffold):
                latent_upscale_nodes["540"] = {
                    "class_type": "LTXVCropGuides",
                    "inputs": {
                        "positive": ltx_positive_ref,
                        "negative": ltx_negative_ref,
                        "latent": refiner_latent_ref,
                    },
                    "_meta": {"title": "Crop LTX Guides Before Upscale Refiner"},
                }
                refiner_positive_ref = ["540", 0]
                refiner_negative_ref = ["540", 1]
                refiner_latent_ref = ["540", 2]
            elif not text_to_video:
                refiner_positive_ref = refiner_positive_ref
                refiner_negative_ref = refiner_negative_ref
            else:
                refiner_positive_ref = ltx_positive_ref
                refiner_negative_ref = ltx_negative_ref
            if first_audio_latent_ref:
                latent_upscale_nodes["30"] = {
                    "class_type": "LTXVConcatAVLatent",
                    "inputs": {"video_latent": refiner_latent_ref, "audio_latent": first_audio_latent_ref},
                    "_meta": {"title": "Recombine AV Latent For Upscale Refiner"},
                }
                refiner_latent_ref = ["30", 0]
            latent_upscale_nodes["26"] = {
                "class_type": "RandomNoise",
                "inputs": {"noise_seed": seed + 1 if seed < 2**32 - 1 else seed},
                "_meta": {"title": "Upscale Refiner Seed"},
            }
            latent_upscale_nodes["27"] = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": LTX_TRANSITION_REFINER_SIGMAS if transition_route else LTX_UPSCALE_REFINER_SIGMAS},
            "_meta": {"title": "Official LTX 2.3 Upscale Refiner Sigmas"},
            }
            refiner_base_model_ref = (
                ltx_model_ref
                if (transition_route and use_ic_timeline_guides and ic_lora_loader_ref)
                else transition_model_ref
            )
            refiner_model_ref = add_detailer_lora(refiner_base_model_ref, "LTX Refiner Detailer LoRA")
            latent_upscale_nodes["28"] = {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": refiner_model_ref,
                    "positive": refiner_positive_ref,
                    "negative": refiner_negative_ref,
                    "cfg": request.cfg,
                },
                "_meta": {"title": "Upscale Refiner CFG Guider"},
            }
            latent_upscale_nodes["31"] = {
                "class_type": "KSamplerSelect",
                "inputs": {"sampler_name": "euler_cfg_pp"},
                "_meta": {"title": "Official LTX Upscale Refiner Sampler Select"},
            }
            latent_upscale_nodes["29"] = {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["26", 0],
                    "guider": ["28", 0],
                    "sampler": ["31", 0],
                    "sigmas": ["27", 0],
                    "latent_image": refiner_latent_ref,
                },
                "_meta": {"title": "LTX Upscale Refiner Sampler"},
            }
            if first_audio_latent_ref:
                latent_upscale_nodes["32"] = {
                    "class_type": "LTXVSeparateAVLatent",
                    "inputs": {"av_latent": ["29", 0]},
                    "_meta": {"title": "Separate Refined AV Latents"},
                }
                decode_latent_ref = ["32", 0]
                first_audio_latent_ref = ["32", 1]
            else:
                decode_latent_ref = ["29", 0]
        else:
            decode_latent_ref = refiner_latent_ref

    if active_audio and first_audio_latent_ref:
        audio_nodes["20"] = {
            "class_type": "LTXVAudioVAEDecode",
            "inputs": {"samples": first_audio_latent_ref, "audio_vae": ["16", 0]},
            "_meta": {"title": "Decode LTX Audio"},
        }
        if audio_volume_normalization_available:
            audio_nodes["33"] = {
                "class_type": "AudioVolumeNormalization",
                "inputs": {"audio": ["20", 0], "target_level": -14.0},
                "_meta": {"title": "Normalize LTX Audio Output"},
            }
            create_video_inputs["audio"] = ["33", 0]
        else:
            create_video_inputs["audio"] = ["20", 0]

    guide_crop_nodes: dict[str, Any] = {}
    if (motion_transfer_route or use_ltx_flfv_reference_guides) and not motion_guides_cropped_before_sampling and not (use_latent_upscale and refine_latent_upscale):
        guide_crop_nodes["81"] = {
            "class_type": "LTXVCropGuides",
            "inputs": {
                "positive": ltx_positive_ref,
                "negative": ltx_negative_ref,
                "latent": decode_latent_ref,
            },
            "_meta": {"title": "Crop LTX Motion Transfer Guides Before Decode"},
        }
        decode_latent_ref = ["81", 2]

    final_scale_nodes: dict[str, Any] = {}
    if use_latent_upscale:
        factor = _ltx_latent_upscale_factor(latent_upscale_name)
        latent_output_width = int(round(width * factor))
        latent_output_height = int(round(height * factor))
        if latent_output_width != final_width or latent_output_height != final_height:
            final_scale_nodes["34"] = {
                "class_type": "ImageScale",
                "inputs": {
                    "image": ["13", 0],
                    "upscale_method": "lanczos",
                    "width": final_width,
                    "height": final_height,
                    "crop": "disabled",
                },
                "_meta": {"title": "Match Requested LTX Output Size"},
            }
            create_video_image_ref = ["34", 0]
            create_video_inputs["images"] = create_video_image_ref

    trim_nodes: dict[str, Any] = {}
    if not text_to_video and length > 1:
        trim_nodes["36"] = {
            "class_type": "VHS_SelectImages",
            "inputs": {
                "image": create_video_image_ref,
                "indexes": f"0:{length}",
                "err_if_missing": False,
                "err_if_empty": True,
            },
            "_meta": {"title": "Trim LTX Frames To Requested Duration"},
        }
        create_video_image_ref = ["36", 0]
        create_video_inputs["images"] = create_video_image_ref

    sigma_node: dict[str, Any]
    if motion_transfer_route and motion_control_mode == "camera":
        sigma_node = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": LTX_DISTILLED_8_STEP_SIGMAS},
            "_meta": {"title": "Official LTX 2.3 Cameraman Sigmas"},
        }
    elif steps == 8:
        sigma_node = {
            "class_type": "ManualSigmas",
            "inputs": {"sigmas": LTX_TRANSITION_8_STEP_SIGMAS if transition_route else LTX_DISTILLED_8_STEP_SIGMAS},
            "_meta": {"title": "Official LTX 2.3 8-Step Sigmas"},
        }
    else:
        sigma_node = {
            "class_type": "LTXVScheduler",
            "inputs": {
                "steps": steps,
                "max_shift": float(video_options.get("max_shift") or 2.05),
                "base_shift": float(video_options.get("base_shift") or 0.95),
                "stretch": True,
                "terminal": float(video_options.get("terminal") or 0.1),
                "latent": scheduler_latent_ref,
            },
            "_meta": {"title": "LTX Scheduler"},
        }

    video_output_nodes: dict[str, Any]
    if video_combine_node:
        video_output_inputs: dict[str, Any] = {
            "images": create_video_image_ref,
            "frame_rate": float(fps),
            "loop_count": 0,
            "filename_prefix": f"NEXUS_BTA_LTX23_IMG2VID_{final_width}x{final_height}",
            "format": "video/h264-mp4",
            "pix_fmt": "yuv420p",
            "crf": int(_number_or_none(video_options.get("video_crf")) or 16),
            "save_metadata": True,
            "trim_to_audio": False,
            "pingpong": False,
            "save_output": True,
        }
        if "audio" in create_video_inputs:
            video_output_inputs["audio"] = create_video_inputs["audio"]
        video_output_nodes = {
            "15": {
                "class_type": video_combine_node,
                "inputs": video_output_inputs,
                "_meta": {"title": "Save High Quality Video"},
            }
        }
    else:
        video_output_nodes = {
            "14": {
                "class_type": "CreateVideo",
                "inputs": create_video_inputs,
                "_meta": {"title": "Create Video"},
            },
            "15": {
                "class_type": "SaveVideo",
                "inputs": {
                    "video": ["14", 0],
                    "filename_prefix": f"NEXUS_BTA_LTX23_IMG2VID_{final_width}x{final_height}",
                    "format": "mp4",
                    "codec": "h264",
                },
                "_meta": {"title": "Save Video"},
            },
        }

    ltx_model_loader = (
        {
            "class_type": "UnetLoaderGGUF",
            "inputs": {"unet_name": checkpoint_name},
            "_meta": {"title": "Load LTX 2.3 GGUF Model"},
        }
        if checkpoint_name.lower().endswith(".gguf")
        else {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": checkpoint_name},
            "_meta": {"title": "Load LTX 2.3 Checkpoint"},
        }
    )

    workflow = {
        "1": ltx_model_loader,
        "2": {
            "class_type": "LTXAVTextEncoderLoader",
            "inputs": {"text_encoder": text_encoder_name, "ckpt_name": text_projection_name or checkpoint_name, "device": str(video_options.get("text_encoder_device") or "default")},
            "_meta": {"title": "LTX Text Encoder"},
        },
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": reference_image_name},
            "_meta": {"title": "Reference Image"},
        },
        "4": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": positive_prompt},
            "_meta": {"title": "Positive Prompt"},
        },
        "5": {
            "class_type": "CLIPTextEncode",
            "inputs": {"clip": ["2", 0], "text": request.negative_prompt},
            "_meta": {"title": "Negative Prompt"},
        },
        "6": {
            "class_type": "LTXVConditioning",
            "inputs": {"positive": ["4", 0], "negative": ["5", 0], "frame_rate": float(fps)},
            "_meta": {"title": "LTX Frame Rate Conditioning"},
        },
        "7": {
            "class_type": "EmptyLTXVLatentVideo" if (text_to_video or use_first_last_guides or motion_transfer_route or ltx_native_loop_sampler_route) else "LTXVImgToVideo",
            "inputs": (
                {
                    "width": width,
                    "height": height,
                    "length": length,
                    "batch_size": max(1, request.batch_size),
                }
                if (text_to_video or use_first_last_guides or motion_transfer_route or ltx_native_loop_sampler_route)
                else {
                    "positive": ["6", 0],
                    "negative": ["6", 1],
                    "vae": video_vae_ref,
                    "image": reference_image_ref,
                    "width": width,
                    "height": height,
                    "length": length,
                    "batch_size": max(1, request.batch_size),
                    "strength": (
                        float(video_options.get("motion_strength") or (0.30 if loop_uses_start_as_end else request.img2img.denoise) or 0.85)
                    ),
                }
            ),
            "_meta": {"title": "Text To Video Latent" if text_to_video else "Image To Video"},
        },
        "8": {
            "class_type": "RandomNoise",
            "inputs": {"noise_seed": seed},
            "_meta": {"title": "Seed"},
        },
        "9": {
            "class_type": "KSamplerSelect",
            "inputs": {"sampler_name": loop_sampler_name},
            "_meta": {"title": "Sampler"},
        },
        "10": {
            **sigma_node,
        },
        "11": (
            {
                "class_type": "STGGuiderAdvanced",
                "inputs": {
                    "model": main_model_ref,
                    "positive": ltx_positive_ref,
                    "negative": ltx_negative_ref,
                    "skip_steps_sigma_threshold": 0.998,
                    "cfg_star_rescale": True,
                    "sigmas": "1.0, 0.9933, 0.9850, 0.9767, 0.9008, 0.6180",
                    "cfg_values": loop_cfg_values,
                    "stg_scale_values": "1.2, 1.0, 0.8, 0.45, 0.2, 0",
                    "stg_rescale_values": "0.7, 0.7, 0.7, 0.7, 0.7, 0",
                    "stg_layers_indices": "[29], [29], [29], [29], [29], [29]",
                },
                "_meta": {"title": "LTX Native Loop STG Guider"},
            }
            if ltx_native_loop_sampler_route
            else {
                "class_type": "CFGGuider",
                "inputs": {
                    "model": main_model_ref,
                    "positive": ltx_positive_ref,
                    "negative": ltx_negative_ref,
                    "cfg": request.cfg,
                },
                "_meta": {"title": "CFG Guider"},
            }
        ),
        "12": (
            {
                "class_type": "LTXVLoopingSampler",
                "inputs": {
                    "model": main_model_ref,
                    "vae": video_vae_ref,
                    "noise": ["8", 0],
                    "sampler": ["9", 0],
                    "sigmas": ["10", 0],
                    "guider": ["11", 0],
                    "latents": sampler_latent_ref,
                    "temporal_tile_size": int(_number_or_none(video_options.get("ltx_loop_temporal_tile_size")) or 80),
                    "temporal_overlap": int(_number_or_none(video_options.get("ltx_loop_temporal_overlap")) or 24),
                    "guiding_strength": float(_number_or_none(video_options.get("ltx_loop_guiding_strength")) or 0.0),
                    "temporal_overlap_cond_strength": float(_number_or_none(video_options.get("ltx_loop_overlap_strength")) or 0.65),
                    "cond_image_strength": float(_number_or_none(video_options.get("ltx_loop_cond_image_strength")) or 0.72),
                    "horizontal_tiles": int(_number_or_none(video_options.get("ltx_loop_horizontal_tiles")) or 1),
                    "vertical_tiles": int(_number_or_none(video_options.get("ltx_loop_vertical_tiles")) or 1),
                    "spatial_overlap": int(_number_or_none(video_options.get("ltx_loop_spatial_overlap")) or 1),
                    "optional_cond_images": ["3", 0],
                    "optional_cond_image_indices": "0",
                    "adain_factor": float(_number_or_none(video_options.get("ltx_loop_adain_factor")) or 0.12),
                    "optional_normalizing_latents": sampler_latent_ref,
                },
                "_meta": {"title": "LTX Native Looping Sampler"},
            }
            if ltx_native_loop_sampler_route
            else {
                "class_type": "SamplerCustomAdvanced",
                "inputs": {
                    "noise": ["8", 0],
                    "guider": ["11", 0],
                    "sampler": ["9", 0],
                    "sigmas": ["10", 0],
                    "latent_image": sampler_latent_ref,
                },
                "_meta": {"title": "LTX Sampler"},
            }
        ),
        "13": {
            "class_type": "LTXVTiledVAEDecode",
            "inputs": {
                "vae": video_vae_ref,
                "latents": decode_latent_ref,
                "horizontal_tiles": 2,
                "vertical_tiles": 2,
                "overlap": 6,
                "last_frame_fix": False,
                "working_device": "auto",
                "working_dtype": "auto",
            },
            "_meta": {"title": "Decode Frames"},
        },
    }
    if use_ltx_nag_model:
        workflow["37"] = {
            "class_type": "LTX2_NAG",
            "inputs": {
                "model": ltx_model_ref,
                "nag_scale": float(_number_or_none(video_options.get("nag_scale")) or 11.0),
                "nag_alpha": float(_number_or_none(video_options.get("nag_alpha")) or 0.25),
                "nag_tau": float(_number_or_none(video_options.get("nag_tau")) or 2.5),
                "nag_cond_video": ltx_negative_ref,
                "inplace": True,
            },
            "_meta": {"title": "Official LTX Cameraman/Transition NAG"},
        }
    if text_to_video:
        workflow.pop("3", None)
    workflow.update(video_output_nodes)
    workflow.update(video_vae_nodes)
    workflow.update(preprocess_nodes)
    workflow.update(audio_nodes)
    workflow.update(guide_crop_nodes)
    workflow.update(latent_upscale_nodes)
    workflow.update(final_scale_nodes)
    workflow.update(trim_nodes)
    workflow.update(ltx_lora_nodes)
    return workflow


def _selected_text(value: Any) -> str:
    text = str(value or "").strip()
    return "" if text.lower() in {"", "automatic", "auto", "none"} else text


def patch_workflow(
    api: dict[str, Any],
    request: GenerateRequest,
    assets: dict[str, Any] | None = None,
) -> dict[str, Any]:
    seed = request.seed if request.seed >= 0 else random.randint(0, 2**32 - 1)
    sampler = normalize_sampler(request.sampler)
    scheduler = normalize_scheduler(request.scheduler)
    assets = assets or {}
    model_name = assets.get("primary_model") or request.model_name or Path(request.model_path or "").name
    video_options = request.video or {}
    director_options = request.director or {}
    preset = request.preset.lower()
    model3d_engine = str((request.model3d or {}).get("engine") or "").lower() if preset == "model3d" else ""
    director_negative_prompt = (
        str(director_options.get("local_negative_prompts") or "").strip()
        if preset == "ltx" and str(request.workspace or "").lower() == "director"
        else ""
    )
    if not director_negative_prompt:
        director_negative_prompt = request.negative_prompt
    effective_prompt = request.prompt
    effective_negative_prompt = director_negative_prompt
    if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting":
        outpaint_prompt = (
            "seamless natural video outpainting, extend the original scene beyond the canvas, "
            "preserve the source video region exactly, continuous lighting and perspective"
        )
        outpaint_negative = (
            "black bars, dark overlay, dark border, dividing line, seam, hard edge, text, letters, "
            "subtitle, watermark, stretched edge, edge streaks, blur, zoom, crop, distorted source"
        )
        effective_prompt = f"{effective_prompt}, {outpaint_prompt}" if str(effective_prompt or "").strip() else outpaint_prompt
        effective_negative_prompt = (
            f"{effective_negative_prompt}, {outpaint_negative}"
            if str(effective_negative_prompt or "").strip()
            else outpaint_negative
        )
    fps_value = _number_or_none(video_options.get("fps"))
    seconds_value = _number_or_none(video_options.get("seconds") or video_options.get("duration"))
    frames_value = _number_or_none(video_options.get("frames"))
    motion_strength_value = _number_or_none(video_options.get("motion_strength"))
    shift_value = _number_or_none(video_options.get("shift"))
    max_shift_value = _number_or_none(video_options.get("max_shift"))
    base_shift_value = _number_or_none(video_options.get("base_shift"))
    terminal_value = _number_or_none(video_options.get("terminal"))
    if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting":
        def nearest_ltx_dimension(value: Any) -> int:
            aligned = round(max(1, int(round(_number_or_none(value) or 512))) / 32) * 32
            return max(64, min(4096, aligned or 512))

        def nearest_ltx_frame_count(value: Any) -> int:
            frames = max(9, int(round(_number_or_none(value) or 9)))
            if (frames - 1) % 8 == 0:
                return frames
            lower = max(9, ((frames - 1) // 8) * 8 + 1)
            upper = max(9, math.ceil((frames - 1) / 8) * 8 + 1)
            return lower if (frames - lower) <= (upper - frames) else upper

        def ltx_outpaint_sigmas(steps: int) -> str:
            base = [1.0, 0.99375, 0.9875, 0.98125, 0.975, 0.909375, 0.725, 0.421875, 0.0]
            step_count = max(1, int(steps or 4))
            if step_count == len(base) - 1:
                return ", ".join(f"{value:g}" for value in base)
            sampled: list[float] = []
            scale = (len(base) - 1) / step_count
            for index in range(step_count + 1):
                position = index * scale
                lo = int(position)
                hi = min(len(base) - 1, lo + 1)
                frac = position - lo
                sampled.append(base[lo] + (base[hi] - base[lo]) * frac)
            return ", ".join(f"{value:.6g}" for value in sampled)

        def ltx_outpaint_fit_size() -> tuple[int, int] | None:
            source_width = _number_or_none(video_options.get("outpaint_source_width"))
            source_height = _number_or_none(video_options.get("outpaint_source_height"))
            if not source_width or not source_height or source_width <= 0 or source_height <= 0:
                return None
            scale = min(request.width / source_width, request.height / source_height)
            fit_width = nearest_ltx_dimension(source_width * scale)
            fit_height = nearest_ltx_dimension(source_height * scale)
            if fit_width > request.width:
                fit_width = max(32, (int(request.width) // 32) * 32)
                fit_height = nearest_ltx_dimension(fit_width * source_height / source_width)
            if fit_height > request.height:
                fit_height = max(32, (int(request.height) // 32) * 32)
                fit_width = nearest_ltx_dimension(fit_height * source_width / source_height)
            return max(32, int(fit_width)), max(32, int(fit_height))

        request.width = nearest_ltx_dimension(request.width)
        request.height = nearest_ltx_dimension(request.height)
        if frames_value is not None:
            frames_value = nearest_ltx_frame_count(frames_value)
        elif seconds_value and fps_value:
            frames_value = nearest_ltx_frame_count(seconds_value * fps_value)
    elif preset == "ltx" and seconds_value and fps_value:
        frames_value = max(1, round(seconds_value * fps_value))

    positive_patched = False
    lora_slot = 0
    aspect_gcd = math.gcd(max(1, int(request.width or 1)), max(1, int(request.height or 1)))
    target_aspect_w = max(1, int(request.width or 1) // max(1, aspect_gcd))
    target_aspect_h = max(1, int(request.height or 1) // max(1, aspect_gcd))

    def ltx_director_base_dimension(value: Any) -> int:
        numeric = max(0, int(round(_number_or_none(value) or 0)))
        if not numeric:
            return numeric
        latent_choice = str(video_options.get("latent_upscale") or "").strip().lower()
        latent_enabled = assets.get("latent_upscale") and latent_choice not in {"none", "off", "false", "0", "no"}
        if not latent_enabled:
            return numeric
        half = max(64, numeric // 2)
        return max(64, half - (half % 32))

    def set_input_or_linked(inputs: dict[str, Any], key: str, value: Any) -> None:
        current = inputs.get(key)
        if isinstance(current, list) and current:
            source = api.get(str(current[0]))
            if isinstance(source, dict):
                source_inputs = source.setdefault("inputs", {})
                if isinstance(source_inputs, dict):
                    for constant_key in ("value", "number", "int", "float"):
                        if constant_key in source_inputs:
                            source_inputs[constant_key] = value
                            return
        inputs[key] = value

    def patch_side_menu_asset_inputs(node: dict[str, Any], inputs: dict[str, Any], class_lower: str, title: str) -> None:
        haystack = " ".join([class_lower, title]).lower()
        if "vae_name" in inputs:
            if "audio" in haystack and assets.get("audio_vae"):
                inputs["vae_name"] = assets["audio_vae"]
            elif ("preview" in haystack or "tae" in haystack) and assets.get("preview_vae"):
                inputs["vae_name"] = assets["preview_vae"]
            elif ("video" in haystack or "ltx" in haystack) and assets.get("video_vae"):
                inputs["vae_name"] = assets["video_vae"]
            elif assets.get("vae"):
                inputs["vae_name"] = assets["vae"]
            elif assets.get("video_vae"):
                inputs["vae_name"] = assets["video_vae"]
        if "text_encoder" in inputs and assets.get("text_encoder"):
            inputs["text_encoder"] = assets["text_encoder"]
        if preset == "ltx" and class_lower == "ltxavtextencoderloader" and "device" in inputs:
            inputs["device"] = str(video_options.get("text_encoder_device") or "default")
        if "clip_name1" in inputs and assets.get("flux_clip_l"):
            inputs["clip_name1"] = assets["flux_clip_l"]
        if "clip_name2" in inputs and assets.get("text_encoder"):
            inputs["clip_name2"] = assets["text_encoder"]
        if "clip_name" in inputs:
            if ("clip_l" in haystack or "clip-l" in haystack) and assets.get("flux_clip_l"):
                inputs["clip_name"] = assets["flux_clip_l"]
            elif assets.get("text_encoder"):
                inputs["clip_name"] = assets["text_encoder"]
            if assets.get("text_encoder"):
                text_encoder_name = str(assets["text_encoder"])
                if preset == "anima" and is_anima_qwen35_text_encoder(text_encoder_name):
                    node["class_type"] = "LoadQwen35AnimaCLIP"
                    inputs.pop("type", None)
                    inputs.pop("device", None)
                    inputs.setdefault("use_calibration", False)
                    inputs.setdefault("use_alignment", False)
                    inputs.setdefault("alignment_strength", 0.0)
                    inputs.setdefault("output_scale", 1.0)
                elif preset == "anima" and class_lower == "loadqwen35animaclip":
                    node["class_type"] = "CLIPLoader"
                    inputs["type"] = "stable_diffusion"
                    inputs["device"] = str(video_options.get("text_encoder_device") or "default")
                    inputs.pop("use_calibration", None)
                    inputs.pop("use_alignment", None)
                    inputs.pop("alignment_strength", None)
                    inputs.pop("output_scale", None)
                elif text_encoder_name.lower().endswith(".gguf"):
                    node["class_type"] = "CLIPLoaderGGUF"
                    inputs.pop("device", None)
                elif class_lower == "cliploadergguf":
                    node["class_type"] = "CLIPLoader"
                    inputs.setdefault("device", str(video_options.get("text_encoder_device") or "default"))
        if "unet_name" in inputs and assets.get("primary_model"):
            if preset == "wan":
                if "high" in haystack and assets.get("wan_high_model"):
                    inputs["unet_name"] = assets["wan_high_model"]
                elif "low" in haystack and assets.get("wan_low_model"):
                    inputs["unet_name"] = assets["wan_low_model"]
                else:
                    inputs["unet_name"] = assets["primary_model"]
            else:
                inputs["unet_name"] = assets["primary_model"]
        if "model_name" in inputs:
            if "upscale" in haystack and assets.get("latent_upscale"):
                inputs["model_name"] = assets["latent_upscale"]
            elif assets.get("primary_model"):
                inputs["model_name"] = assets["primary_model"]

    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type", ""))
        class_lower = class_type.lower()
        inputs = node.setdefault("inputs", {})
        title = str(node.get("_meta", {}).get("title", "")).lower()

        if class_lower == "ltxdirector" and director_options:
            timeline_data = director_options.get("timeline_data_json")
            if not timeline_data and isinstance(director_options.get("timeline_data"), dict):
                timeline_data = json.dumps(director_options["timeline_data"])
            director_patch = {
                "global_prompt": effective_prompt,
                "duration_frames": director_options.get("duration_frames"),
                "duration_seconds": director_options.get("duration_seconds"),
                "timeline_data": timeline_data,
                "local_prompts": director_options.get("local_prompts"),
                "segment_lengths": director_options.get("segment_lengths"),
                "epsilon": director_options.get("epsilon"),
                "guide_strength": director_options.get("guide_strength"),
                "use_custom_audio": director_options.get("use_custom_audio"),
                "frame_rate": director_options.get("frame_rate") or fps_value,
                "display_mode": director_options.get("display_mode"),
                "custom_width": ltx_director_base_dimension(director_options.get("custom_width") or request.width),
                "custom_height": ltx_director_base_dimension(director_options.get("custom_height") or request.height),
                "resize_method": director_options.get("resize_method"),
                "divisible_by": director_options.get("divisible_by"),
                "img_compression": director_options.get("img_compression"),
            }
            for key, value in director_patch.items():
                if value is not None and value != "":
                    inputs[key] = value

        if model_name:
            for key in ["ckpt_name", "unet_name"]:
                if key in inputs:
                    if preset == "ltx" and class_lower == "ltxavtextencoderloader" and key == "ckpt_name":
                        continue
                    if preset == "wan" and key == "unet_name":
                        wan_haystack = " ".join([title, class_lower, str(inputs.get(key) or "")]).lower()
                        if "high" in wan_haystack and assets.get("wan_high_model"):
                            inputs[key] = assets["wan_high_model"]
                        elif "low" in wan_haystack and assets.get("wan_low_model"):
                            inputs[key] = assets["wan_low_model"]
                        else:
                            inputs[key] = model_name
                        continue
                    inputs[key] = model_name
            if "unet_name" in inputs:
                if str(model_name).lower().endswith(".gguf"):
                    node["class_type"] = "UnetLoaderGGUF"
                    inputs.pop("weight_dtype", None)
                elif class_lower == "unetloadergguf":
                    node["class_type"] = "UNETLoader"
                    inputs.setdefault("weight_dtype", "default")

        if preset == "ltx" and class_lower == "ltxavtextencoderloader":
            if assets.get("text_encoder"):
                inputs["text_encoder"] = assets["text_encoder"]
            if assets.get("text_projection"):
                inputs["ckpt_name"] = assets["text_projection"]

        patch_side_menu_asset_inputs(node, inputs, class_lower, title)

        if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" and "value" in inputs:
            if "target_aspect_w" in title:
                inputs["value"] = target_aspect_w
            elif "target_aspect_h" in title:
                inputs["value"] = target_aspect_h
        if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" and class_lower == "imagepadkj":
            set_input_or_linked(inputs, "target_width", request.width)
            set_input_or_linked(inputs, "target_height", request.height)
            outpaint_pad = video_options.get("outpaint_pad") if isinstance(video_options, dict) else {}
            if isinstance(outpaint_pad, dict):
                for side in ("left", "right", "top", "bottom"):
                    if side in inputs:
                        set_input_or_linked(inputs, side, max(0, int(round(_number_or_none(outpaint_pad.get(side)) or 0))))
            inputs["color"] = "0, 0, 0"
            inputs["pad_mode"] = str(video_options.get("outpaint_pad_mode") or "edge")

        for key, value in list(inputs.items()):
            if isinstance(value, str) and _looks_like_model_file(value):
                if class_lower == "ltxavtextencoderloader" and key == "ckpt_name":
                    continue
                replacement, lora_slot = _replacement_for_model_input(
                    key=key,
                    value=value,
                    class_type=class_type,
                    title=title,
                    assets=assets,
                    lora_slot=lora_slot,
                )
                if replacement:
                    inputs[key] = replacement

        if "text" in inputs and ("textencode" in class_lower or "conditioning" in class_lower or "prompt" in title):
            is_negative = "negative" in title or "negative" in class_lower
            if is_negative:
                inputs["text"] = effective_negative_prompt
            elif not positive_patched:
                inputs["text"] = effective_prompt
                positive_patched = True
        elif "text" in inputs:
            if "negative" in title or "negative" in class_lower:
                inputs["text"] = effective_negative_prompt
            elif "positive" in title or "positive" in class_lower:
                inputs["text"] = effective_prompt
        if "prompt" in inputs and ("textencode" in class_lower or "conditioning" in class_lower or "prompt" in title):
            is_negative = "negative" in title or "negative" in class_lower
            if is_negative:
                inputs["prompt"] = effective_negative_prompt
            elif not positive_patched:
                inputs["prompt"] = effective_prompt
                positive_patched = True
            elif "positive" in title:
                inputs["prompt"] = effective_prompt

        for key in ["width", "empty_latent_width"]:
            if key in inputs:
                set_input_or_linked(inputs, key, request.width)
        for key in ["height", "empty_latent_height"]:
            if key in inputs:
                set_input_or_linked(inputs, key, request.height)
        for key in ["seed", "noise_seed"]:
            if key in inputs:
                if (
                    preset == "ltx"
                    and key == "noise_seed"
                    and ("refiner" in title or "upscale" in title)
                    and isinstance(inputs.get(key), (int, float))
                    and int(inputs[key]) != int(seed)
                ):
                    continue
                set_input_or_linked(inputs, key, seed)
        if "steps" in inputs:
            if preset == "ltx":
                step_value = max(1, int(request.steps or 4)) if request.workflow_id == "ltx23-video-outpainting" else max(8, int(request.steps or 8))
            elif preset == "wan":
                step_value = 4
            else:
                step_value = request.steps
            set_input_or_linked(inputs, "steps", step_value)
        if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" and class_lower == "manualsigmas":
            inputs["sigmas"] = ltx_outpaint_sigmas(max(1, int(request.steps or 4)))
        if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" and class_lower == "resizeimagemasknode":
            if inputs.get("resize_type") == "scale by multiplier" and "resize_type.multiplier" in inputs:
                set_input_or_linked(inputs, "resize_type.multiplier", 1.0)
        if preset == "flux" and "guidance" in inputs:
            set_input_or_linked(inputs, "guidance", request.cfg)
        if "cfg" in inputs:
            set_input_or_linked(inputs, "cfg", 1.0 if preset == "flux" else request.cfg)
        if "sampler_name" in inputs:
            if preset == "ltx" and ("refiner" in title or "upscale" in title):
                inputs["sampler_name"] = "euler_cfg_pp"
                continue
            inputs["sampler_name"] = sampler
        if "scheduler" in inputs:
            inputs["scheduler"] = scheduler
        if "denoise" in inputs:
            inputs["denoise"] = request.img2img.denoise if request.activity == "img2img" else request.denoise
        if assets.get("mask_image") and "image" in inputs and ("mask" in title or "mask" in class_lower):
            inputs["image"] = assets["mask_image"]
        elif (
            preset == "ltx"
            and request.workflow_id == "ltx23-video-outpainting"
            and assets.get("outpaint_reference_image")
            and class_lower == "loadimage"
            and "image" in inputs
        ):
            inputs["image"] = assets["outpaint_reference_image"]
        elif (
            assets.get("reference_image")
            and "image" in inputs
            and ("loadimage" in class_lower or ("image" in title and model3d_engine != "triposplat"))
        ):
            inputs["image"] = assets["reference_image"]
        if "batch_size" in inputs:
            batch_size_value = 1 if preset == "qwen" and request.activity == "img2img" else max(1, request.batch_size)
            set_input_or_linked(inputs, "batch_size", batch_size_value)
        is_video_loader = class_lower in {"loadvideo", "vhs_loadvideo", "loadvideoui"} or "load video" in title
        if assets.get("base_video") and is_video_loader and "video" in inputs:
            inputs["video"] = assets["base_video"]
        if assets.get("base_video") and is_video_loader and "force_rate" in inputs and fps_value is not None:
            set_input_or_linked(inputs, "force_rate", fps_value)
        if assets.get("base_video") and is_video_loader and "frame_load_cap" in inputs and frames_value is not None:
            set_input_or_linked(inputs, "frame_load_cap", max(1, int(round(frames_value))))
        if assets.get("base_video") and is_video_loader and "skip_first_frames" in inputs:
            set_input_or_linked(inputs, "skip_first_frames", max(0, int(round(_number_or_none(video_options.get("base_start_frame")) or 0))))
        if assets.get("base_video") and is_video_loader and "custom_width" in inputs:
            fit_size = ltx_outpaint_fit_size() if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" else None
            set_input_or_linked(
                inputs,
                "custom_width",
                fit_size[0] if fit_size else (0 if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" else request.width),
            )
        if assets.get("base_video") and is_video_loader and "custom_height" in inputs:
            fit_size = ltx_outpaint_fit_size() if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" else None
            set_input_or_linked(
                inputs,
                "custom_height",
                fit_size[1] if fit_size else (0 if preset == "ltx" and request.workflow_id == "ltx23-video-outpainting" else request.height),
            )
        if assets.get("base_video") and "start_frame" in inputs:
            set_input_or_linked(inputs, "start_frame", max(0, int(round(_number_or_none(video_options.get("base_start_frame")) or 0))))
        if assets.get("base_video") and "end_frame" in inputs and frames_value is not None:
            set_input_or_linked(inputs, "end_frame", max(1, int(round(frames_value))))
        for key in ["fps", "frame_rate", "framerate"]:
            if key in inputs and fps_value is not None:
                set_input_or_linked(inputs, key, fps_value)
        for key in ["seconds", "duration", "duration_seconds", "video_seconds"]:
            if key in inputs and seconds_value is not None:
                if key == "duration" and class_lower in {"loadaudioui", "emptyaudio", "trimaudioduration"}:
                    continue
                set_input_or_linked(inputs, key, seconds_value)
        for key in ["frames", "num_frames", "frame_count", "frames_number", "length"]:
            if key in inputs and frames_value is not None:
                set_input_or_linked(inputs, key, max(1, round(frames_value)))
        for key in ["motion_strength", "strength"]:
            if key in inputs and motion_strength_value is not None and ("motion" in title or "video" in title or "ltx" in class_lower or "i2v" in title):
                set_input_or_linked(inputs, key, motion_strength_value)
        if "shift" in inputs and shift_value is not None:
            set_input_or_linked(inputs, "shift", shift_value)
        if "max_shift" in inputs and max_shift_value is not None:
            set_input_or_linked(inputs, "max_shift", max_shift_value)
        if "base_shift" in inputs and base_shift_value is not None:
            set_input_or_linked(inputs, "base_shift", base_shift_value)
        if "terminal" in inputs and terminal_value is not None:
            set_input_or_linked(inputs, "terminal", terminal_value)

    _ensure_external_vae_loader(api, assets)
    _ensure_model3d_trellis_route(api, request, assets)
    _ensure_model3d_triposplat_route(api, request, assets)
    _ensure_ltx_outpaint_attention_mask(api, request)
    if not (preset == "ltx" and request.workflow_id == "ltx23-video-outpainting"):
        _apply_side_menu_loras(api, request)
    _ensure_zimage_reference_route(api, request, assets)
    _ensure_qwen_image_edit_route(api, request, assets)
    _ensure_qwen_multiview_route(api, request, assets)
    _ensure_qwen_multi_reference_route(api, request, assets)
    _ensure_wan_start_end_frame_route(api, request, assets)
    _ensure_img2img_reference_resize_routes(api, request)
    _ensure_controlnet_route(api, request, assets)
    _ensure_ltx_workflow_extensions(api, request, assets)
    _ensure_inpaint_mask_route(api, request, assets)
    _ensure_ltx_director_frame_trim(api, request)
    return api


def _ensure_ltx_outpaint_attention_mask(api: dict[str, Any], request: GenerateRequest) -> None:
    if request.preset.lower() != "ltx" or request.workflow_id != "ltx23-video-outpainting":
        return
    pad_node_id = None
    guide_node_id = None
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower == "imagepadkj":
            pad_node_id = str(node_id)
        elif class_lower in {"ltxaddvideoicloraguide", "ltxaddvideoicloraguideadvanced"}:
            guide_node_id = str(node_id)
    if not pad_node_id or not guide_node_id:
        return

    invert_node_id = str(_next_api_node_id(api))
    api[invert_node_id] = {
        "class_type": "InvertMask",
        "inputs": {"mask": [pad_node_id, 1]},
        "_meta": {"title": "LTX Outpaint Preserve Source Mask"},
    }
    guide = api.get(guide_node_id)
    if not isinstance(guide, dict):
        return
    guide["class_type"] = "LTXAddVideoICLoRAGuideAdvanced"
    inputs = guide.setdefault("inputs", {})
    if isinstance(inputs, dict):
        inputs["attention_strength"] = float(request.video.get("outpaint_attention_strength") or 1.0)
        inputs["attention_mask"] = [invert_node_id, 0]


def _model3d_number(options: dict[str, Any], key: str, default: float) -> float:
    value = _number_or_none(options.get(key))
    return default if value is None else value


def _model3d_int(options: dict[str, Any], key: str, default: int) -> int:
    return int(round(_model3d_number(options, key, float(default))))


def _model3d_bool(options: dict[str, Any], key: str, default: bool = False) -> bool:
    value = options.get(key)
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on", "enabled"}


def _model3d_low_vram_enabled(options: dict[str, Any], request: GenerateRequest) -> bool:
    if "low_vram" in options:
        return _model3d_bool(options, "low_vram", True)
    runtime = getattr(request, "runtime", None)
    policy = str(getattr(runtime, "vram_policy", "") or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    limit = _number_or_none(getattr(runtime, "gpu_memory_gb", None))
    if policy not in {"gpu", "gpuonly", "onlygpu", "cudaonly"}:
        return True
    if limit is not None and limit <= 14:
        return True
    return False


def _model3d_keep_models_loaded(options: dict[str, Any], request: GenerateRequest) -> bool:
    if "keep_models_loaded" in options:
        return _model3d_bool(options, "keep_models_loaded", False)
    runtime = getattr(request, "runtime", None)
    policy = str(getattr(runtime, "vram_policy", "") or "").strip().lower().replace("_", "").replace("-", "").replace(" ", "")
    limit = _number_or_none(getattr(runtime, "gpu_memory_gb", None))
    if policy in {"gpu", "gpuonly", "onlygpu", "cudaonly"} and limit is not None and limit >= 16:
        return True
    return False


def _trellis2_model_name(value: Any = None) -> str:
    allowed = {
        "microsoft/trellis.2-4b": "microsoft/TRELLIS.2-4B",
        "visualbruno/trellis.2-4b-fp8": "visualbruno/TRELLIS.2-4B-FP8",
        "tencentarc/pixal3d-t": "TencentARC/Pixal3D-T",
    }
    key = str(value or "").strip().replace("\\", "/").lower()
    if key in allowed:
        return allowed[key]
    if key.endswith("/trellis.2-4b") or key == "trellis.2-4b":
        return "microsoft/TRELLIS.2-4B"
    if key.endswith("/trellis.2-4b-fp8") or key == "trellis.2-4b-fp8":
        return "visualbruno/TRELLIS.2-4B-FP8"
    if key.endswith("/pixal3d-t") or key == "pixal3d-t":
        return "TencentARC/Pixal3D-T"
    return "microsoft/TRELLIS.2-4B"


def _model3d_sparse_resolution(options: dict[str, Any]) -> int:
    explicit = _number_or_none(options.get("sparse_structure_resolution"))
    if explicit is not None:
        return max(32, min(128, int(round(explicit))))
    voxel = _model3d_int(options, "voxel_resolution", 1024)
    return 32 if voxel <= 1024 else 64


def _model3d_max_tokens(options: dict[str, Any], request: GenerateRequest) -> int:
    explicit = _number_or_none(options.get("max_num_tokens") or options.get("max_tokens"))
    if explicit is not None:
        return max(50000, min(999999, int(round(explicit))))
    runtime = getattr(request, "runtime", None)
    limit = _number_or_none(getattr(runtime, "gpu_memory_gb", None))
    target_faces = _model3d_int(options, "decimation_target", 250000)
    if limit is not None and limit <= 12:
        return 49152
    if limit is not None and limit <= 16:
        return max(49152, min(98304, int(target_faces)))
    return 999999


def _model3d_skip_shape_cascade(options: dict[str, Any], request: GenerateRequest) -> bool:
    if "skip_shape_cascade" in options:
        return _model3d_bool(options, "skip_shape_cascade", False)
    runtime = getattr(request, "runtime", None)
    limit = _number_or_none(getattr(runtime, "gpu_memory_gb", None))
    if limit is not None and limit <= 12 and _trellis2_attention_backend(options, request) == "sdpa":
        return True
    return False


def _model3d_sampler(options: dict[str, Any], key: str, default: str = "euler") -> str:
    value = str(options.get(key) or default).strip().lower()
    return value if value in {"euler", "heun", "rk4", "rk5"} else default


def _module_available(name: str) -> bool:
    try:
        if name == "xformers":
            import xformers.ops  # noqa: F401

            return importlib.util.find_spec("xformers._C") is not None
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def _trellis2_attention_backend(options: dict[str, Any], request: GenerateRequest | None = None) -> str:
    requested = str(options.get("trellis_backend") or options.get("attention_backend") or "").lower()
    runtime = getattr(request, "runtime", None) if request is not None else None
    runtime_attention = str(getattr(runtime, "attention_backend", "") or "").lower().replace("_", "").replace("-", "").replace(" ", "")
    xformers_disabled = bool(getattr(runtime, "disable_xformers", False))
    if requested in {"flash_attn", "xformers", "sdpa", "flash_attn_3"}:
        if requested == "flash_attn" and not _module_available("flash_attn"):
            return "sdpa"
        if requested == "flash_attn_3" and not _module_available("flash_attn_interface"):
            return "sdpa"
        if requested == "xformers" and (xformers_disabled or not _module_available("xformers")):
            return "sdpa"
        return requested
    if runtime_attention in {"pytorch", "pytorchsdpa", "sdpa"}:
        return "sdpa"
    if runtime_attention in {"flash", "flashattention"} and _module_available("flash_attn"):
        return "flash_attn"
    if not xformers_disabled and _module_available("xformers"):
        return "xformers"
    if _module_available("flash_attn"):
        return "flash_attn"
    if _module_available("flash_attn_interface"):
        return "flash_attn_3"
    return "sdpa"


def _trellis2_sparse_backend(options: dict[str, Any], request: GenerateRequest | None = None) -> str:
    requested = str(options.get("trellis_sparse_backend") or options.get("sparse_backend") or "").lower()
    runtime = getattr(request, "runtime", None) if request is not None else None
    xformers_disabled = bool(getattr(runtime, "disable_xformers", False))
    if requested == "xformers" and not xformers_disabled and _module_available("xformers"):
        return "xformers"
    if requested == "flash_attn" and _module_available("flash_attn"):
        return "flash_attn"
    if not xformers_disabled and _module_available("xformers"):
        return "xformers"
    return "flash_attn"


def _trellis2_seed(seed: int) -> int:
    value = int(seed)
    return min(value, 0x7FFFFFFF) if value >= 0 else random.randint(0, 0x7FFFFFFF)


def _set_existing_model3d_inputs(inputs: dict[str, Any], names: list[str], value: Any) -> None:
    if value is None:
        return
    for name in names:
        if name in inputs:
            inputs[name] = value


def _set_model3d_inputs(inputs: dict[str, Any], names: list[str], value: Any) -> None:
    if value is None:
        return
    matched = False
    for name in names:
        if name in inputs:
            inputs[name] = value
            matched = True
    if not matched and names:
        inputs[names[0]] = value


def _find_trellis_loader(api: dict[str, Any], title_token: str) -> str | None:
    wanted = title_token.lower()
    fallback: str | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "trellis2loadimagewithtransparency":
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if wanted in title:
            return str(node_id)
        fallback = fallback or str(node_id)
    return fallback if wanted == "front" else None


def _replace_model3d_refs(value: Any, node_id: str, replacements: dict[int, list[Any]]) -> Any:
    if isinstance(value, list):
        if len(value) == 2 and str(value[0]) == node_id and isinstance(value[1], int) and value[1] in replacements:
            return list(replacements[value[1]])
        return [_replace_model3d_refs(item, node_id, replacements) for item in value]
    if isinstance(value, dict):
        return {key: _replace_model3d_refs(item, node_id, replacements) for key, item in value.items()}
    return value


def _bypass_model3d_shape_cascade(api: dict[str, Any]) -> None:
    cascade_id: str | None = None
    shape_source_id: str | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "trellis2shapecascademultiviewgenerator":
            continue
        shape_ref = node.get("inputs", {}).get("shape_slat") if isinstance(node.get("inputs"), dict) else None
        if isinstance(shape_ref, list) and shape_ref:
            cascade_id = str(node_id)
            shape_source_id = str(shape_ref[0])
            break
    if not cascade_id or not shape_source_id:
        return
    replacements = {
        0: [shape_source_id, 0],
        1: [shape_source_id, 1],
        2: [shape_source_id, 2],
        3: [shape_source_id, 3],
    }
    for node_id, node in list(api.items()):
        if str(node_id) == cascade_id or not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            node["inputs"] = _replace_model3d_refs(inputs, cascade_id, replacements)
    api.pop(cascade_id, None)


def _ensure_trellis_reference_preprocess(api: dict[str, Any], image_name: str, view: str) -> list[Any]:
    loader_id = _find_trellis_loader(api, view)
    if not loader_id:
        loader_id = str(_next_api_node_id(api))
        api[loader_id] = {
            "class_type": "Trellis2LoadImageWithTransparency",
            "inputs": {},
            "_meta": {"title": f"{view.title()} Image"},
        }
    loader = api[str(loader_id)]
    inputs = loader.setdefault("inputs", {})
    if isinstance(inputs, dict):
        inputs["image"] = image_name
    loader.setdefault("_meta", {})["title"] = f"{view.title()} Image"

    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "trellis2preprocessimage":
            continue
        inputs = node.setdefault("inputs", {})
        if isinstance(inputs, dict) and inputs.get("image") in ([str(loader_id), 0], [str(loader_id), 2]):
            inputs["image"] = [str(loader_id), 2]
            return [str(node_id), 0]

    preprocess_id = str(_next_api_node_id(api))
    api[preprocess_id] = {
        "class_type": "Trellis2PreProcessImage",
        "inputs": {"image": [str(loader_id), 2], "padding": 25, "remove_background": True, "max_size": 1024},
        "_meta": {"title": f"Preprocess {view.title()} Image"},
    }
    return [preprocess_id, 0]


def _ensure_model3d_trellis_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    if str(request.preset or "").lower() != "model3d":
        return
    options = request.model3d or {}
    if str(options.get("engine") or "trellis2").lower() == "triposplat":
        return
    raw_refs = assets.get("reference_images") or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    refs = [str(item) for item in raw_refs if str(item or "").strip()][:4]
    if not refs and assets.get("reference_image"):
        refs = [str(assets["reference_image"])]
    if not refs:
        return

    skip_shape_cascade = _model3d_skip_shape_cascade(options, request)
    low_vram_profile = _model3d_low_vram_enabled(options, request)
    think_steps = max(1, _model3d_int(options, "think_steps", 14))
    guidance = _model3d_number(options, "guidance", 7.5)
    sparse_steps = max(1, _model3d_int(options, "sparse_steps", think_steps))
    shape_steps = max(1, _model3d_int(options, "shape_steps", think_steps))
    material_steps = max(1, _model3d_int(options, "material_steps", min(think_steps, 16)))
    if (skip_shape_cascade or low_vram_profile) and _model3d_bool(options, "fast_low_vram", False):
        sparse_steps = min(sparse_steps, 8)
        shape_steps = min(shape_steps, 8)
        material_steps = min(material_steps, 8)
    sparse_guidance = _model3d_number(options, "sparse_guidance", guidance)
    shape_guidance = _model3d_number(options, "shape_guidance", guidance)
    target_faces = max(100000, _model3d_int(options, "decimation_target", 250000))
    max_num_tokens = _model3d_max_tokens(options, request)
    requested_resolution = _model3d_int(options, "voxel_resolution", 1024)
    shape_resolution = 512 if skip_shape_cascade else requested_resolution
    remesh_resolution = 512 if skip_shape_cascade else (1024 if requested_resolution <= 1024 else 2048)

    view_order = ["front", "left", "right", "back"]
    image_refs: dict[str, list[Any]] = {}
    for view, image_name in zip(view_order, refs):
        image_refs[view] = _ensure_trellis_reference_preprocess(api, image_name, view)

    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        title = str(node.get("_meta", {}).get("title", "")).lower()
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue

        if class_lower == "trellis2imagecondmultiviewgenerator":
            for view in view_order:
                key = f"{view}_image"
                if view in image_refs:
                    inputs[key] = image_refs[view]
                elif key != "front_image":
                    inputs.pop(key, None)
        elif class_lower == "trellis2loadmodel":
            model_name = _trellis2_model_name(options.get("model") or inputs.get("modelname"))
            inputs["modelname"] = model_name
            inputs.pop("model", None)
            inputs["backend"] = _trellis2_attention_backend(options, request)
            inputs["device"] = "cuda"
            inputs["low_vram"] = _model3d_low_vram_enabled(options, request)
            inputs["keep_models_loaded"] = _model3d_keep_models_loaded(options, request)
            inputs.setdefault("conv_backend", "flex_gemm")
            inputs["sparse_backend"] = _trellis2_sparse_backend(options, request)
            inputs.setdefault("use_reconviagen", False)
        elif class_lower == "trellis2preprocessimage":
            inputs["padding"] = _model3d_int(options, "preprocess_padding", 25)
            inputs["remove_background"] = _model3d_bool(options, "remove_background", True)
            inputs["max_size"] = requested_resolution
            _set_existing_model3d_inputs(inputs, ["mask", "texture_mask", "inpaint_mask"], assets.get("mask_image"))
        elif class_lower == "trellis2sparsemultiviewgenerator":
            inputs["seed"] = _trellis2_seed(request.seed)
            inputs["sparse_structure_steps"] = sparse_steps
            inputs["sparse_structure_guidance_strength"] = sparse_guidance
            inputs["sparse_structure_guidance_rescale"] = _model3d_number(options, "sparse_rescale", 0.7)
            inputs["sparse_structure_rescale_t"] = _model3d_number(options, "sparse_rescale_t", 5)
            inputs["sparse_structure_resolution"] = _model3d_sparse_resolution(options)
            inputs["fill_holes"] = True
            inputs["hole_fill_algorithm"] = "flood_fill"
            inputs["keep_only_shell"] = True
            _set_model3d_inputs(inputs, ["sparse_structure_sampler", "sampler"], _model3d_sampler(options, "sparse_sampler"))
            _set_existing_model3d_inputs(inputs, ["dino_lock"], _model3d_number(options, "dino_lock", 0.0))
            _set_existing_model3d_inputs(inputs, ["dino_substeps"], _model3d_int(options, "dino_substeps", 4))
            _set_existing_model3d_inputs(inputs, ["dino_foundation_cap"], _model3d_number(options, "dino_foundation_cap", 1.0))
        elif class_lower == "trellis2shapemultiviewgenerator":
            _set_model3d_inputs(inputs, ["resolution"], shape_resolution)
            _set_model3d_inputs(inputs, ["shape_steps", "steps"], shape_steps)
            _set_model3d_inputs(inputs, ["shape_guidance_strength", "guidance_strength"], shape_guidance)
            _set_model3d_inputs(inputs, ["shape_guidance_rescale", "guidance_rescale"], _model3d_number(options, "shape_rescale", 0.5))
            _set_model3d_inputs(inputs, ["shape_rescale_t", "rescale_t"], _model3d_number(options, "shape_rescale_t", 3))
            _set_model3d_inputs(inputs, ["shape_sampler", "sampler"], _model3d_sampler(options, "shape_sampler"))
            _set_model3d_inputs(inputs, ["shape_guidance_interval_start", "threshold"], 0.1)
            _set_model3d_inputs(inputs, ["shape_guidance_interval_end", "batch_size"], 1)
        elif class_lower == "trellis2shapecascademultiviewgenerator":
            if "seed" in inputs:
                inputs["seed"] = _trellis2_seed(request.seed)
            _set_model3d_inputs(inputs, ["to_resolution", "resolution"], shape_resolution)
            inputs["sparse_structure_resolution"] = _model3d_sparse_resolution(options)
            _set_model3d_inputs(inputs, ["max_num_tokens"], max_num_tokens)
            _set_model3d_inputs(inputs, ["shape_steps", "steps"], shape_steps)
            _set_model3d_inputs(inputs, ["shape_guidance_strength", "guidance_strength"], shape_guidance)
            _set_model3d_inputs(inputs, ["shape_guidance_rescale", "guidance_rescale"], _model3d_number(options, "shape_rescale", 0.5))
            _set_model3d_inputs(inputs, ["shape_rescale_t", "rescale_t"], _model3d_number(options, "shape_rescale_t", 3))
            _set_model3d_inputs(inputs, ["shape_sampler", "sampler"], _model3d_sampler(options, "shape_sampler"))
            _set_model3d_inputs(inputs, ["shape_guidance_interval_start"], 0.1)
            _set_model3d_inputs(inputs, ["shape_guidance_interval_end"], 1)
        elif class_lower == "trellis2texslatmultiviewgenerator":
            inputs["resolution"] = 512 if skip_shape_cascade else (1024 if _model3d_int(options, "texture_size", 1024) >= 1024 else 512)
            _set_model3d_inputs(inputs, ["texture_steps", "steps"], material_steps)
            inputs["texture_guidance_strength"] = _model3d_number(options, "material_guidance", 1.0)
            inputs["texture_guidance_rescale"] = _model3d_number(options, "material_rescale", 0.0)
            inputs["texture_rescale_t"] = _model3d_number(options, "material_rescale_t", 3)
            _set_model3d_inputs(inputs, ["texture_sampler", "sampler"], _model3d_sampler(options, "texture_sampler"))
            _set_existing_model3d_inputs(inputs, ["texture_sharp", "sharp", "sharpness"], _model3d_number(options, "texture_sharp", 3.0))
            _set_existing_model3d_inputs(inputs, ["texture_details", "details", "detail"], _model3d_number(options, "texture_details", 1.0))
            _set_existing_model3d_inputs(inputs, ["prompt", "positive", "text", "texture_prompt"], str(options.get("texture_positive") or request.prompt or ""))
            _set_existing_model3d_inputs(inputs, ["negative", "negative_prompt"], str(options.get("texture_negative") or request.negative_prompt or ""))
            _set_existing_model3d_inputs(inputs, ["denoise", "denoise_strength", "strength"], _model3d_number(options, "texture_denoise", 0.45))
            _set_existing_model3d_inputs(inputs, ["mask_grow", "mask_expand", "grow_mask"], _model3d_int(options, "texture_mask_grow", 8))
            _set_existing_model3d_inputs(inputs, ["mask", "texture_mask", "inpaint_mask"], assets.get("mask_image"))
            _set_existing_model3d_inputs(inputs, ["albedo", "albedo_mode"], _model3d_bool(options, "texture_albedo_mode", True))
            _set_existing_model3d_inputs(inputs, ["partial_regenerate", "partially_regenerate", "use_existing_texture"], _model3d_bool(options, "texture_partial_regenerate", True))
            _set_existing_model3d_inputs(inputs, ["ignore_geometry", "geometry_guidance"], _model3d_bool(options, "texture_ignore_geometry", False))
            node.setdefault("_meta", {})["nexus_texture_paint"] = {
                "mode": str(options.get("texture_mode") or ""),
                "generation": str(options.get("texture_generation") or ""),
                "output_map": str(options.get("texture_output_map") or ""),
                "control_type": str(options.get("texture_control_type") or ""),
                "preserve_existing": _model3d_bool(options, "texture_preserve_existing", True),
                "mask_space": str(options.get("texture_mask_space") or "uv"),
                "active_tool": str(options.get("texture_active_tool") or ""),
                "active_layer": str(options.get("texture_active_layer") or ""),
                "active_camera": str(options.get("texture_active_camera") or ""),
                "cameras": options.get("texture_cameras") if isinstance(options.get("texture_cameras"), list) else [],
                "layers": options.get("texture_layers") if isinstance(options.get("texture_layers"), list) else [],
                "mask_image": bool(assets.get("mask_image")),
            }
        elif class_lower == "trellis2meshwithvoxelmultiviewgenerator":
            for view in view_order:
                key = f"{view}_image"
                if view in image_refs:
                    inputs[key] = image_refs[view]
                elif key != "front_image":
                    inputs.pop(key, None)
            inputs["seed"] = _trellis2_seed(request.seed)
            if skip_shape_cascade:
                default_pipeline_type = "512"
            elif low_vram_profile and not _model3d_bool(options, "high_quality_low_vram", False):
                default_pipeline_type = "1024" if requested_resolution >= 1024 else "512"
            else:
                default_pipeline_type = "1024_cascade" if requested_resolution >= 1024 else "512"
            inputs["pipeline_type"] = str(options.get("pipeline_type") or default_pipeline_type)
            inputs["sparse_structure_steps"] = sparse_steps
            inputs["sparse_structure_guidance_strength"] = sparse_guidance
            inputs["sparse_structure_guidance_rescale"] = _model3d_number(options, "sparse_rescale", 0.7)
            inputs["sparse_structure_rescale_t"] = _model3d_number(options, "sparse_rescale_t", 5)
            inputs["shape_steps"] = shape_steps
            inputs["shape_guidance_strength"] = shape_guidance
            inputs["shape_guidance_rescale"] = _model3d_number(options, "shape_rescale", 0.5)
            inputs["shape_rescale_t"] = _model3d_number(options, "shape_rescale_t", 3)
            inputs["texture_steps"] = 1
            inputs["texture_guidance_strength"] = 1
            inputs["texture_guidance_rescale"] = 0
            inputs["texture_rescale_t"] = 3
            inputs["max_num_tokens"] = max_num_tokens
            inputs["sparse_structure_resolution"] = _model3d_sparse_resolution(options)
            inputs["generate_texture_slat"] = False
            inputs["sparse_structure_guidance_interval_start"] = 0.1
            inputs["sparse_structure_guidance_interval_end"] = 1
            inputs["shape_guidance_interval_start"] = 0.1
            inputs["shape_guidance_interval_end"] = 1
            inputs["texture_guidance_interval_start"] = 0
            inputs["texture_guidance_interval_end"] = 0.9
            inputs["use_tiled_decoder"] = True
            inputs["front_axis"] = str(options.get("front_axis") or "z")
            inputs["blend_temperature"] = _model3d_number(options, "blend_temperature", 1)
            inputs["sampler"] = _model3d_sampler(options, "shape_sampler")
            inputs["fill_holes"] = True
            inputs["hole_iterations"] = _model3d_int(options, "hole_iterations", 1)
            inputs["hole_fill_algorithm"] = "flood_fill"
            inputs["keep_only_shell"] = True
            inputs["verbose"] = False
            inputs["dino_lock"] = _model3d_number(options, "dino_lock", 0.0)
            inputs["dino_substeps"] = _model3d_int(options, "dino_substeps", 4)
            inputs["dino_foundation_cap"] = _model3d_number(options, "dino_foundation_cap", 1.0)
        elif class_lower == "trellis2meshtexturingmultiview":
            for view in view_order:
                key = f"{view}_image"
                if view in image_refs:
                    inputs[key] = image_refs[view]
                elif key != "front_image":
                    inputs.pop(key, None)
            inputs["seed"] = _trellis2_seed(request.seed)
            inputs["texture_steps"] = material_steps
            inputs["texture_guidance_strength"] = _model3d_number(options, "material_guidance", 3.0)
            inputs["texture_guidance_rescale"] = _model3d_number(options, "material_rescale", 0.2)
            inputs["texture_rescale_t"] = _model3d_number(options, "material_rescale_t", 3)
            inputs["resolution"] = 512 if _model3d_low_vram_enabled(options, request) else min(1536, max(512, _model3d_int(options, "texture_projection_resolution", 1024)))
            inputs["texture_size"] = _model3d_int(options, "texture_size", 1024)
            inputs["texture_alpha_mode"] = "OPAQUE"
            inputs["double_side_material"] = _model3d_bool(options, "double_side_material", False)
            inputs["texture_guidance_interval_start"] = 0
            inputs["texture_guidance_interval_end"] = 0.9
            inputs["bake_on_vertices"] = _model3d_bool(options, "bake_on_vertices", False)
            inputs["use_custom_normals"] = _model3d_bool(options, "use_custom_normals", False)
            inputs["mesh_cluster_threshold_cone_half_angle_rad"] = _model3d_number(options, "mesh_cluster_angle", 60)
            inputs["front_axis"] = str(options.get("front_axis") or "z")
            inputs["blend_temperature"] = _model3d_number(options, "blend_temperature", 1)
            inputs["sampler"] = _model3d_sampler(options, "texture_sampler")
            inputs["inpainting"] = str(options.get("texture_inpainting") or "telea")
            inputs["verbose"] = False
            inputs["dino_lock"] = _model3d_number(options, "dino_lock", 0.0)
            inputs["dino_substeps"] = _model3d_int(options, "dino_substeps", 4)
            inputs["dino_foundation_cap"] = _model3d_number(options, "dino_foundation_cap", 1.0)
        elif class_lower == "trellis2exportmesh":
            inputs["file_format"] = "glb"
        elif class_lower == "trellis2reconstructmeshwithquad":
            inputs["remesh_band"] = _model3d_number(options, "remesh_band", 1.0)
            inputs["resolution"] = remesh_resolution
            inputs["remove_floaters"] = True
            inputs["remove_inner_faces"] = True
        elif class_lower == "trellis2remeshwithquad":
            inputs["remesh_band"] = _model3d_number(options, "remesh_band", 1.0)
            inputs["remesh_project"] = _model3d_number(options, "remesh_project", 0.0)
            inputs["dual_contouring_resolution"] = str(remesh_resolution)
            inputs["remove_floaters"] = True
            inputs["remove_inner_faces"] = True
        elif class_lower == "trellis2fillholeswithcumesh":
            inputs["max_permieters"] = _model3d_number(options, "max_hole_perimeter", 1.0)
        elif class_lower == "trellis2simplifymesh":
            inputs["method"] = "Cumesh"
            _set_existing_model3d_inputs(inputs, ["target_face_num"], target_faces)
        elif class_lower == "trellis2unwrapandrasterizer":
            inputs.setdefault("reorient_vertices", "90 degrees")
        elif class_lower == "trellis2meshwithvoxeltotrimesh":
            inputs["reorient_vertices"] = "90 degrees"
        elif class_lower == "primitiveint":
            if "target face" in title:
                inputs["value"] = target_faces
            elif "texture" in title:
                inputs["value"] = _model3d_int(options, "texture_size", 1024)
        elif "texture" in class_lower or "inpaint" in class_lower or "project" in class_lower:
            _set_existing_model3d_inputs(inputs, ["prompt", "positive", "text", "texture_prompt"], str(options.get("texture_positive") or request.prompt or ""))
            _set_existing_model3d_inputs(inputs, ["negative", "negative_prompt"], str(options.get("texture_negative") or request.negative_prompt or ""))
            _set_existing_model3d_inputs(inputs, ["mask", "texture_mask", "inpaint_mask"], assets.get("mask_image"))
            _set_existing_model3d_inputs(inputs, ["denoise", "denoise_strength", "strength"], _model3d_number(options, "texture_denoise", 0.45))
            _set_existing_model3d_inputs(inputs, ["albedo", "albedo_mode"], _model3d_bool(options, "texture_albedo_mode", True))
    if skip_shape_cascade:
        _bypass_model3d_shape_cascade(api)


def _ensure_model3d_triposplat_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    if str(request.preset or "").lower() != "model3d":
        return
    options = request.model3d or {}
    if str(options.get("engine") or "").lower() != "triposplat":
        return
    raw_refs = assets.get("reference_images") or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    image_name = next((str(item) for item in raw_refs if str(item or "").strip()), "") or str(assets.get("reference_image") or "")
    if not image_name:
        return

    num_gaussians = max(32768, min(1048576, _model3d_int(options, "tripo_num_gaussians", 262144)))
    runtime = getattr(request, "runtime", None)
    limit = _number_or_none(getattr(runtime, "gpu_memory_gb", None))
    if limit is not None and limit <= 12:
        num_gaussians = min(num_gaussians, 262144)
    steps = max(8, min(40, _model3d_int(options, "tripo_steps", 20)))
    cfg = max(1.0, min(8.0, _model3d_number(options, "tripo_cfg", 3.0)))
    render_size = max(512, min(1536, _model3d_int(options, "tripo_render_size", 1024)))
    if limit is not None and limit <= 12:
        render_size = min(render_size, 1024)
    remove_background = _model3d_bool(options, "tripo_remove_background", True)
    export_glb = _model3d_bool(options, "tripo_export_glb", True)
    delete_nodes: set[str] = set()
    load_image_id: str | None = None
    conditioning_vae_ids: set[str] = set()
    decoder_vae_ids: set[str] = set()
    splat_file_ids: set[str] = set()
    splat_mesh_ids: set[str] = set()

    for scan_node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        scan_node_id_text = str(scan_node_id)
        class_lower = str(node.get("class_type", "")).lower()
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        if class_lower == "triposplatconditioning":
            vae_ref = inputs.get("vae")
            if isinstance(vae_ref, list) and vae_ref:
                conditioning_vae_ids.add(str(vae_ref[0]))
        elif class_lower == "vaedecodetriposplat":
            vae_ref = inputs.get("vae")
            if isinstance(vae_ref, list) and vae_ref:
                decoder_vae_ids.add(str(vae_ref[0]))
        elif class_lower == "splattofile3d":
            splat_file_ids.add(scan_node_id_text)
        elif class_lower == "splattomesh":
            splat_mesh_ids.add(scan_node_id_text)

    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        node_id_text = str(node_id)
        class_lower = str(node.get("class_type", "")).lower()
        title = str(node.get("_meta", {}).get("title", "")).lower()
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if class_lower == "loadimage":
            inputs["image"] = image_name
            inputs["upload"] = "image"
            load_image_id = node_id_text
        elif class_lower == "ksampler":
            inputs["seed"] = _trellis2_seed(request.seed)
            inputs["steps"] = steps
            inputs["cfg"] = cfg
            inputs["sampler_name"] = str(options.get("tripo_sampler") or "dpmpp_2m")
            inputs["scheduler"] = str(options.get("tripo_scheduler") or "simple")
            inputs["denoise"] = 1
        elif class_lower == "unetloader":
            inputs["unet_name"] = str(options.get("tripo_model") or "triposplat_fp16.safetensors")
            inputs["weight_dtype"] = "default"
        elif class_lower == "clipvisionloader":
            inputs["clip_name"] = str(options.get("tripo_clip_vision") or "dino_v3_vit_h.safetensors")
        elif class_lower == "vaeloader":
            if node_id_text in conditioning_vae_ids:
                inputs["vae_name"] = "flux2-vae.safetensors"
            elif node_id_text in decoder_vae_ids:
                inputs["vae_name"] = "triposplat_vae_decoder_fp16.safetensors"
            elif "flux" in title or str(inputs.get("vae_name") or "").lower().startswith("flux"):
                inputs["vae_name"] = "flux2-vae.safetensors"
            else:
                inputs["vae_name"] = "triposplat_vae_decoder_fp16.safetensors"
        elif class_lower == "loadbackgroundremovalmodel":
            inputs["bg_removal_name"] = "birefnet.safetensors"
        elif class_lower == "comfyswitchnode":
            if "mask source" in title:
                inputs["switch"] = bool(remove_background)
            else:
                inputs["switch"] = False
        elif class_lower == "triposplatpreprocessimage":
            _set_existing_model3d_inputs(inputs, ["num_gaussians", "sampling_num_gaussians", "num_gaussians_1"], num_gaussians)
        elif class_lower == "vaedecodetriposplat":
            _set_model3d_inputs(inputs, ["num_gaussians", "sampling_num_gaussians", "num_gaussians_1"], num_gaussians)
        elif class_lower == "rendersplat":
            _set_model3d_inputs(inputs, ["width"], render_size)
            _set_model3d_inputs(inputs, ["height"], render_size)
            _set_existing_model3d_inputs(inputs, ["frames"], _model3d_int(options, "tripo_preview_frames", 24))
            _set_existing_model3d_inputs(inputs, ["fov"], _model3d_number(options, "tripo_fov", 75))
            inputs.pop("camera_info", None)
        elif class_lower == "createcamerainfo":
            delete_nodes.add(node_id_text)
            inputs["mode"] = "orbit"
            inputs["mode.yaw"] = _model3d_number(options, "tripo_camera_yaw", 35)
            inputs["mode.pitch"] = _model3d_number(options, "tripo_camera_pitch", 30)
            inputs["mode.distance"] = _model3d_number(options, "tripo_camera_distance", 2.5)
            inputs["target_x"] = 0
            inputs["target_y"] = 0
            inputs["target_z"] = 0
            inputs["roll"] = 0
            inputs["fov"] = _model3d_number(options, "tripo_fov", 75)
            inputs["zoom"] = 1
            inputs["camera_type"] = "perspective"
        elif class_lower == "createvideo":
            delete_nodes.add(node_id_text)
        elif class_lower == "splattofile3d":
            inputs["format"] = "spz"
        elif class_lower == "saveglb":
            mesh_ref = inputs.get("mesh")
            if isinstance(mesh_ref, list) and mesh_ref and str(mesh_ref[0]) in splat_file_ids:
                inputs["filename_prefix"] = "3D/TripoSplat_Splat"
            elif isinstance(mesh_ref, list) and mesh_ref and str(mesh_ref[0]) in splat_mesh_ids:
                inputs["filename_prefix"] = "3D/TripoSplat_Mesh"
            else:
                inputs["filename_prefix"] = "3D/TripoSplat_Model"
        elif class_lower == "savevideo":
            delete_nodes.add(node_id_text)
        elif class_lower == "previewimage":
            delete_nodes.add(node_id_text)
        elif class_lower == "splattomesh":
            _set_existing_model3d_inputs(inputs, ["resolution"], _model3d_int(options, "tripo_mesh_resolution", 256 if limit is not None and limit <= 12 else 384))
            _set_existing_model3d_inputs(inputs, ["kernel"], _model3d_int(options, "tripo_mesh_kernel", 5))
            _set_existing_model3d_inputs(inputs, ["smooth"], _model3d_int(options, "tripo_mesh_smooth", 2))
            _set_existing_model3d_inputs(inputs, ["level"], _model3d_number(options, "tripo_mesh_level", 0.4))
            _set_existing_model3d_inputs(inputs, ["min_component"], _model3d_int(options, "tripo_mesh_min_component", 500))
            _set_existing_model3d_inputs(inputs, ["min_opacity"], _model3d_number(options, "tripo_mesh_min_opacity", 0.02))
            _set_existing_model3d_inputs(inputs, ["color_sharpen"], _model3d_number(options, "tripo_mesh_color_sharpen", 2.0))

    if not export_glb:
        for node_id, node in api.items():
            if not isinstance(node, dict):
                continue
            if str(node.get("class_type", "")).lower() == "splattomesh":
                delete_nodes.add(str(node_id))
    if delete_nodes:
        for node_id, node in list(api.items()):
            if not isinstance(node, dict):
                continue
            inputs = node.get("inputs")
            if not isinstance(inputs, dict):
                continue
            class_lower = str(node.get("class_type", "")).lower()
            if class_lower == "saveglb":
                mesh_ref = inputs.get("mesh")
                if isinstance(mesh_ref, list) and mesh_ref and str(mesh_ref[0]) in delete_nodes:
                    delete_nodes.add(str(node_id))
                    continue
            for key, value in list(inputs.items()):
                if isinstance(value, list) and value and str(value[0]) in delete_nodes:
                    inputs.pop(key, None)
    for node_id in delete_nodes:
        api.pop(node_id, None)
    if load_image_id:
        for node in api.values():
            if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "invertmask":
                continue
            inputs = node.setdefault("inputs", {})
            if isinstance(inputs, dict) and "mask" not in inputs:
                inputs["mask"] = [load_image_id, 1]


def _ensure_ltx_director_frame_trim(api: dict[str, Any], request: GenerateRequest) -> None:
    if request.preset.lower() != "ltx" or str(request.workspace or "").lower() != "director":
        return
    director_options = request.director or {}
    video_options = request.video or {}
    target_frames_value = _number_or_none(director_options.get("duration_frames"))
    if target_frames_value is not None:
        target_frames = max(1, int(round(target_frames_value)) + 1)
    else:
        target_frames = max(1, int(round(_number_or_none(video_options.get("frames")) or 0)))
    if target_frames <= 1:
        return
    indexes = f"0:{target_frames}"
    for _node_id, node in list(api.items()):
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() not in {"createvideo", "vhs_videocombine"}:
            continue
        inputs = node.setdefault("inputs", {})
        image_ref = inputs.get("images") or inputs.get("image")
        if not isinstance(image_ref, list) or not image_ref:
            continue
        existing = api.get(str(image_ref[0]))
        if isinstance(existing, dict) and str(existing.get("class_type", "")).lower() == "vhs_selectimages":
            existing_inputs = existing.setdefault("inputs", {})
            existing_inputs["indexes"] = indexes
            existing_inputs["err_if_missing"] = False
            existing_inputs["err_if_empty"] = True
            return
        trim_id = str(_next_api_node_id(api))
        api[trim_id] = {
            "class_type": "VHS_SelectImages",
            "inputs": {
                "image": image_ref,
                "indexes": indexes,
                "err_if_missing": False,
                "err_if_empty": True,
            },
            "_meta": {"title": "Trim Director Frames To Timeline"},
        }
        if "images" in inputs:
            inputs["images"] = [trim_id, 0]
        else:
            inputs["image"] = [trim_id, 0]
        return


def _ensure_zimage_reference_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    if request.activity != "img2img" or request.preset.lower() not in {"zimageturbo", "zimage"}:
        return
    reference_image = str(assets.get("reference_image") or "").strip()
    if not reference_image:
        return
    vae_ref = _find_vae_ref(api)
    if not vae_ref:
        return
    loader_id = _find_reference_image_node_id(api, reference_image) or _add_load_image_node(api, reference_image, "Reference Image")
    if not loader_id:
        return
    api[str(loader_id)].setdefault("inputs", {})["image"] = reference_image
    api[str(loader_id)].setdefault("_meta", {})["title"] = "Reference Image"

    sampler_id = _find_sampler_node_id(api)
    if sampler_id and isinstance(api.get(sampler_id), dict):
        sampler_inputs = api[sampler_id].setdefault("inputs", {})
        encode_id: str | None = None
        for node_id, node in api.items():
            if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "vaeencode":
                continue
            inputs = node.get("inputs", {})
            if isinstance(inputs, dict) and inputs.get("pixels") == [str(loader_id), 0]:
                encode_id = str(node_id)
                break
        if not encode_id:
            encode_id = str(_next_api_node_id(api))
            api[encode_id] = {
                "class_type": "VAEEncode",
                "inputs": {"pixels": [str(loader_id), 0], "vae": vae_ref},
                "_meta": {"title": "Encode Z-Image Reference"},
            }
        _ensure_encode_reference_resize(api, request, encode_id)
        sampler_inputs["latent_image"] = [encode_id, 0]
        sampler_inputs["denoise"] = request.img2img.denoise
        positive_ref = sampler_inputs.get("positive")
    else:
        positive_ref = None

    positive_node: dict[str, Any] | None = None
    if isinstance(positive_ref, list) and positive_ref:
        candidate = api.get(str(positive_ref[0]))
        if isinstance(candidate, dict):
            positive_node = candidate
    if positive_node is None:
        for node in api.values():
            if not isinstance(node, dict):
                continue
            title = str(node.get("_meta", {}).get("title", "")).lower()
            class_lower = str(node.get("class_type", "")).lower()
            if "negative" in title:
                continue
            if "textencodezimageomni" in class_lower or "cliptextencode" in class_lower:
                positive_node = node
                break
    if positive_node is None:
        return
    inputs = positive_node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        return
    positive_node["class_type"] = "TextEncodeZImageOmni"
    prompt = inputs.pop("text", None)
    inputs["prompt"] = str(prompt if prompt not in {None, ""} else request.prompt)
    inputs["auto_resize_images"] = True
    inputs["vae"] = vae_ref
    inputs["image1"] = [str(loader_id), 0]


def _ensure_qwen_image_edit_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    if request.activity != "img2img" or request.preset.lower() != "qwen":
        return
    if _qwen_pose_studio_handoff(request):
        return
    reference_image = str(assets.get("reference_image") or "").strip()
    if not reference_image:
        return
    raw_refs = assets.get("reference_images") or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    refs = [str(name) for name in raw_refs if str(name or "").strip()][:3]
    multi_reference = len(refs) > 1
    use_qwen_plus_route = multi_reference
    qwen_linear_view = _truthy_option((request.video or {}).get("qwen_linear_view"))

    loader_id = _find_qwen_reference_loader_id(api, 1, reference_image) or _add_load_image_node(api, reference_image, "Reference Image 1")
    if not loader_id:
        return
    loader = api.get(str(loader_id))
    if isinstance(loader, dict):
        loader.setdefault("inputs", {})["image"] = reference_image
        loader.setdefault("_meta", {})["title"] = "Reference Image 1"
        scaled_loader_ref = _ensure_image_ref_scaled(
        api,
        [str(loader_id), 0],
        request.width,
        request.height,
        "Resize QWEN Reference 1 To Side Menu",
    ) or [str(loader_id), 0]
    qwen_base_ref = scaled_loader_ref

    vae_ref = _find_vae_ref(api)
    positive_node_id: str | None = None
    negative_node_id: str | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower not in {"textencodeqwenimageedit", "textencodeqwenimageeditplus"}:
            continue
        node["class_type"] = "TextEncodeQwenImageEditPlus" if use_qwen_plus_route else "TextEncodeQwenImageEdit"
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        prompt_text = inputs.pop("text", None)
        if prompt_text is not None and "prompt" not in inputs:
            inputs["prompt"] = prompt_text
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if "prompt" not in inputs:
            inputs["prompt"] = request.negative_prompt if "negative" in title else request.prompt
        if vae_ref and "vae" not in inputs:
            inputs["vae"] = vae_ref
        if "negative" in title:
            for key in ("image", "image1", "image2", "image3"):
                inputs.pop(key, None)
        elif use_qwen_plus_route:
            inputs.pop("image", None)
            inputs["image1"] = qwen_base_ref
        else:
            for key in ("image1", "image2", "image3"):
                inputs.pop(key, None)
            inputs["image"] = qwen_base_ref
        if "negative" in title:
            negative_node_id = str(node_id)
        else:
            positive_node_id = str(node_id)

    if _uses_inpaint_mask_mode(request):
        return
    sampler_id = _find_sampler_node_id(api)
    if not sampler_id:
        return
    latent_id: str | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if class_lower in {"vaeencode", "emptyqwenimagelayeredlatentimage"} and (
            "qwen" in title or "reference" in title or "base" in title or "latent" in title
        ):
            latent_id = str(node_id)
            break
    if not latent_id:
        latent_id = str(_next_api_node_id(api))
        api[latent_id] = {
            "class_type": "VAEEncode",
            "inputs": {},
            "_meta": {"title": "QWEN Linear Encode Base Reference" if qwen_linear_view else "Encode QWEN Base Reference"},
        }
    else:
        api[latent_id]["class_type"] = "VAEEncode"
        api[latent_id].setdefault("_meta", {})["title"] = (
            "QWEN Linear Encode Base Reference" if qwen_linear_view else "Encode QWEN Base Reference"
        )
    latent_inputs = api[latent_id].setdefault("inputs", {})
    latent_inputs.clear()
    latent_inputs["pixels"] = qwen_base_ref
    if vae_ref:
        latent_inputs["vae"] = vae_ref
    sampler_inputs = api[sampler_id].setdefault("inputs", {})
    sampler_inputs["latent_image"] = [latent_id, 0]
    if not use_qwen_plus_route:
        model_loader_ref = _find_qwen_model_loader_ref(api)
        if model_loader_ref:
            sampler_inputs["model"] = model_loader_ref
    if positive_node_id:
        sampler_inputs["positive"] = [positive_node_id, 0]
    if negative_node_id:
        sampler_inputs["negative"] = [negative_node_id, 0]
    if qwen_linear_view and positive_node_id:
        method_id = _ensure_conditioning_method_node(api, "QWEN Reference Method", [positive_node_id, 0])
        sampler_inputs["positive"] = [method_id, 0]
    if qwen_linear_view and negative_node_id:
        method_id = _ensure_conditioning_method_node(api, "QWEN Negative Reference Method", [negative_node_id, 0])
        sampler_inputs["negative"] = [method_id, 0]


def _ensure_qwen_multi_reference_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    if request.activity != "img2img" or request.preset.lower() != "qwen":
        return
    if _qwen_pose_studio_handoff(request):
        return
    raw_refs = assets.get("reference_images") or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    refs = [str(name) for name in raw_refs if str(name or "").strip()][:3]
    if len(refs) < 2:
        return

    vae_ref = _find_vae_ref(api)
    loader_refs: list[list[Any]] = []
    for index, image_name in enumerate(refs, start=1):
        node_id = _find_qwen_reference_loader_id(api, index, image_name)
        if not node_id:
            node_id = _add_load_image_node(api, image_name, f"Reference Image {index}")
        if not node_id:
            continue
        node = api.get(str(node_id))
        if isinstance(node, dict):
            node.setdefault("inputs", {})["image"] = image_name
            node.setdefault("_meta", {})["title"] = f"Reference Image {index}"
        scaled_ref = _ensure_image_ref_scaled(
            api,
            [str(node_id), 0],
            request.width,
            request.height,
            f"Resize QWEN Reference {index} To Side Menu",
        ) or [str(node_id), 0]
        loader_refs.append(scaled_ref)
    if len(loader_refs) < 2:
        return

    prompt_prefix = "Use Picture 1 as the base reference"
    if len(loader_refs) >= 2:
        prompt_prefix += ", Picture 2 as the Image 2 style/object reference"
    if len(loader_refs) >= 3:
        prompt_prefix += ", and Picture 3 as the Image 3 additional reference"
    prompt_prefix += ". "

    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower not in {"textencodeqwenimageedit", "textencodeqwenimageeditplus"}:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        node["class_type"] = "TextEncodeQwenImageEditPlus"
        prompt_text = inputs.pop("text", None)
        if prompt_text is not None and "prompt" not in inputs:
            inputs["prompt"] = prompt_text
        if "prompt" not in inputs:
            title = str(node.get("_meta", {}).get("title", "")).lower()
            inputs["prompt"] = request.negative_prompt if "negative" in title else request.prompt
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if "negative" not in title:
            prompt = str(inputs.get("prompt") or request.prompt)
            if "picture 1" not in prompt.lower():
                inputs["prompt"] = prompt_prefix + request.prompt
        inputs.pop("image", None)
        if vae_ref and "vae" not in inputs:
            inputs["vae"] = vae_ref
        if "negative" in title:
            for index in range(1, 6):
                inputs.pop(f"image{index}", None)
            continue
        for index, ref in enumerate(loader_refs, start=1):
            inputs[f"image{index}"] = ref


def _ensure_qwen_multiview_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    qwen_multiview = _qwen_multiview_options(request)
    if not qwen_multiview["enabled"]:
        return
    reference_image = str(assets.get("reference_image") or "").strip()
    if not reference_image:
        return
    loader_id = _find_qwen_reference_loader_id(api, 1, reference_image) or _find_reference_image_node_id(api, reference_image)
    if not loader_id:
        loader_id = _add_load_image_node(api, reference_image, "Reference Image 1")
    if not loader_id:
        return
    loader = api.get(str(loader_id))
    if isinstance(loader, dict):
        loader.setdefault("inputs", {})["image"] = reference_image
        loader.setdefault("_meta", {})["title"] = "Reference Image 1"
    scaled_ref = _ensure_image_ref_scaled(
        api,
        [str(loader_id), 0],
        request.width,
        request.height,
        "Resize QWEN Reference 1 To Side Menu",
    ) or [str(loader_id), 0]

    node_id = None
    for existing_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "qwenmultianglecameranode":
            node_id = str(existing_id)
            break
    if not node_id:
        node_id = str(_next_api_node_id(api))
        api[node_id] = {"class_type": "QwenMultiangleCameraNode", "inputs": {}, "_meta": {"title": "Qwen Multiangle Camera"}}
    node = api[node_id]
    node["class_type"] = "QwenMultiangleCameraNode"
    node.setdefault("_meta", {})["title"] = "Qwen Multiangle Camera"
    inputs = node.setdefault("inputs", {})
    if isinstance(inputs, dict):
        inputs["horizontal_angle"] = int(qwen_multiview["horizontal"])
        inputs["vertical_angle"] = int(qwen_multiview["vertical"])
        inputs["zoom"] = float(qwen_multiview["zoom"])
        inputs["default_prompts"] = False
        inputs["camera_view"] = bool(qwen_multiview["camera_view"])
        inputs["image"] = scaled_ref

    for text_node in api.values():
        if not isinstance(text_node, dict):
            continue
        if str(text_node.get("class_type", "")).lower() not in {"textencodeqwenimageedit", "textencodeqwenimageeditplus"}:
            continue
        text_inputs = text_node.setdefault("inputs", {})
        if not isinstance(text_inputs, dict):
            continue
        title = str(text_node.get("_meta", {}).get("title", "")).lower()
        if "negative" in title:
            if "prompt" in text_inputs:
                text_inputs["prompt"] = ""
            elif "text" in text_inputs:
                text_inputs["text"] = ""
            continue
        if "prompt" in text_inputs:
            text_inputs["prompt"] = [str(node_id), 0]
        elif "text" in text_inputs:
            text_inputs["text"] = [str(node_id), 0]


def _ensure_wan_start_end_frame_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, Any]) -> None:
    if request.activity != "img2img" or request.preset.lower() != "wan":
        return
    raw_refs = assets.get("reference_images") or []
    if isinstance(raw_refs, str):
        raw_refs = [raw_refs]
    refs = [str(name) for name in raw_refs if str(name or "").strip()]
    if not refs:
        fallback = str(assets.get("reference_image") or "").strip()
        refs = [fallback] if fallback else []
    if not refs:
        return

    start_loader_id = _find_reference_image_node_id(api, refs[0]) or _add_load_image_node(api, refs[0], "WAN Start Frame")
    if not start_loader_id:
        return
    start_node = api.get(str(start_loader_id))
    if isinstance(start_node, dict):
        start_node.setdefault("inputs", {})["image"] = refs[0]
        start_node.setdefault("_meta", {})["title"] = "WAN Start Frame"
    start_ref = [str(start_loader_id), 0]

    end_ref: list[Any] | None = None
    if len(refs) > 1:
        end_loader_id = _add_load_image_node(api, refs[1], "WAN End Frame")
        if end_loader_id:
            end_ref = [str(end_loader_id), 0]

    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if "wan" not in class_lower and "wan" not in title:
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        if "start_image" in inputs:
            inputs["start_image"] = start_ref
        elif class_lower == "wanimagetovideo":
            inputs["start_image"] = start_ref
        if end_ref and ("firstlast" in class_lower or "first last" in title or "end_image" in inputs):
            inputs["end_image"] = end_ref


def _ensure_img2img_reference_resize_routes(api: dict[str, Any], request: GenerateRequest) -> None:
    if request.activity != "img2img" or request.preset.lower() in {"ltx", "wan"}:
        return
    for node_id, node in list(api.items()):
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower not in {"vaeencode", "vaeencodeforinpaint"}:
            continue
        _ensure_encode_reference_resize(api, request, str(node_id))
        if class_lower == "vaeencodeforinpaint":
            _ensure_inpaint_mask_resize(api, request, node)


def _ensure_encode_reference_resize(api: dict[str, Any], request: GenerateRequest, encode_node_id: str) -> None:
    node = api.get(str(encode_node_id))
    if not isinstance(node, dict):
        return
    inputs = node.setdefault("inputs", {})
    if not isinstance(inputs, dict):
        return
    pixels_ref = inputs.get("pixels")
    scaled_ref = _ensure_image_ref_scaled(api, pixels_ref, request.width, request.height, "Resize Reference To Side Menu")
    if scaled_ref:
        inputs["pixels"] = scaled_ref


def _ensure_inpaint_mask_resize(api: dict[str, Any], request: GenerateRequest, encode_node: dict[str, Any]) -> None:
    inputs = encode_node.get("inputs", {})
    if not isinstance(inputs, dict):
        return
    mask_ref = inputs.get("mask")
    if not isinstance(mask_ref, list) or not mask_ref:
        return
    mask_node = api.get(str(mask_ref[0]))
    if not isinstance(mask_node, dict) or str(mask_node.get("class_type", "")).lower() != "imagetomask":
        return
    mask_inputs = mask_node.setdefault("inputs", {})
    if not isinstance(mask_inputs, dict):
        return
    scaled_ref = _ensure_image_ref_scaled(
        api,
        mask_inputs.get("image"),
        request.width,
        request.height,
        "Resize Inpaint Mask To Side Menu",
        method="nearest-exact",
    )
    if scaled_ref:
        mask_inputs["image"] = scaled_ref


def _ensure_image_ref_scaled(
    api: dict[str, Any],
    image_ref: Any,
    width: int | float,
    height: int | float,
    title: str,
    *,
    method: str = "lanczos",
) -> list[Any] | None:
    if not isinstance(image_ref, list) or not image_ref:
        return None
    source_id = str(image_ref[0])
    source = api.get(source_id)
    if not isinstance(source, dict):
        return None
    source_class = str(source.get("class_type", "")).lower()
    target_width = max(16, int(width))
    target_height = max(16, int(height))
    if source_class == "imagescale":
        source_inputs = source.setdefault("inputs", {})
        if isinstance(source_inputs, dict):
            source_inputs["width"] = target_width
            source_inputs["height"] = target_height
            source_inputs.setdefault("upscale_method", method)
            source_inputs.setdefault("crop", "disabled")
        source.setdefault("_meta", {})["title"] = title
        return [source_id, 0]
    if source_class != "loadimage":
        return None
    scale_id = str(_next_api_node_id(api))
    api[scale_id] = _image_scale_node([source_id, 0], target_width, target_height, method=method)
    api[scale_id]["_meta"]["title"] = title
    return [scale_id, 0]


def _ensure_qwen_flux_image_ref(api: dict[str, Any], image_ref: Any, title: str) -> list[Any] | None:
    if not isinstance(image_ref, list) or not image_ref:
        return None
    source_id = str(image_ref[0])
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "fluxkontextimagescale":
            continue
        inputs = node.setdefault("inputs", {})
        if isinstance(inputs, dict) and inputs.get("image") == image_ref:
            node.setdefault("_meta", {})["title"] = title
            return [str(node_id), 0]
    if source_id not in api:
        return None
    flux_id = str(_next_api_node_id(api))
    api[flux_id] = _qwen_flux_image_scale_node(image_ref, title)
    return [flux_id, 0]


def _ensure_external_vae_loader(api: dict[str, Any], assets: dict[str, str]) -> None:
    vae_name = assets.get("vae")
    if not vae_name:
        return

    vae_refs: set[tuple[Any, ...]] = set()
    for node in api.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            key_lower = str(key).lower()
            if key_lower != "vae" or not isinstance(value, list) or not value:
                continue
            source = api.get(str(value[0]))
            source_class = str(source.get("class_type", "")).lower() if isinstance(source, dict) else ""
            if source_class != "vaeloader":
                vae_refs.add(tuple(value))
    if not vae_refs:
        return

    loader_id = None
    for candidate_id, node in api.items():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "vaeloader":
            continue
        inputs = node.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if inputs.get("vae_name") == vae_name or "side menu" in title or "vae" in title:
            inputs["vae_name"] = vae_name
            loader_id = str(candidate_id)
            break
    if loader_id is None:
        loader_id = str(_next_api_node_id(api))
        api[loader_id] = {
            "class_type": "VAELoader",
            "inputs": {"vae_name": vae_name},
            "_meta": {"title": "Side Menu VAE Override"},
        }
    replacement = [loader_id, 0]
    for node in api.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs")
        if not isinstance(inputs, dict):
            continue
        for key, value in list(inputs.items()):
            if "vae" in str(key).lower() and isinstance(value, list) and tuple(value) in vae_refs:
                inputs[key] = replacement


def _apply_side_menu_loras(api: dict[str, Any], request: GenerateRequest) -> None:
    selections = _active_lora_selections(request)
    is_ltx = request.preset.lower() == "ltx"
    checkpoint_name = request.model_name or Path(request.model_path or "").name
    flux_model_only_lora = request.preset.lower() == "flux" and _flux_family_from_name(checkpoint_name).startswith("flux2")
    model_only_lora = request.preset.lower() in {"ltx", "qwen", "wan", "zimageturbo", "zimage"} or flux_model_only_lora
    audio_value = (request.director or {}).get("use_custom_audio")
    if audio_value is None:
        audio_value = (request.video or {}).get("active_audio")
    if isinstance(audio_value, str):
        audio_active = audio_value.lower() not in {"false", "0", "off", "none", "no"}
    else:
        audio_active = bool(audio_value)

    def ltx_audio_strength(strength: float) -> float:
        return max(0.0, min(1.0, abs(float(strength)))) if audio_active else 0.0

    def patch_ltx2_lora_inputs(inputs: dict[str, Any], strength_model: float) -> None:
        routed_strength = max(0.0, min(1.0, abs(float(strength_model))))
        inputs["video"] = routed_strength
        inputs["video_to_audio"] = ltx_audio_strength(strength_model)
        inputs["audio"] = ltx_audio_strength(strength_model)
        inputs["audio_to_video"] = ltx_audio_strength(strength_model)
        inputs["other"] = routed_strength

    def effective_strength(lora_name: str, strength_model: float) -> float:
        if is_ltx:
            return _effective_ltx_lora_strength(checkpoint_name, lora_name, strength_model)
        return strength_model

    def protected_ltx_lora_node(node: dict[str, Any]) -> bool:
        if not is_ltx:
            return False
        class_lower = str(node.get("class_type", "")).lower()
        inputs = node.get("inputs") if isinstance(node.get("inputs"), dict) else {}
        title = str(node.get("_meta", {}).get("title", "")).lower()
        lora_name = str(inputs.get("lora_name") or "").lower()
        haystack = " ".join([class_lower, title, lora_name])
        return (
            class_lower == "ltxicloraloadermodelonly"
            or "ic-lora" in haystack
            or "union-control" in haystack
            or "detailer" in haystack
            or "id lora" in haystack
            or "distilled" in haystack
        )

    def pop_selection_for_node(remaining: list[tuple[str, float, float]], node: dict[str, Any]) -> tuple[str, float, float] | None:
        if not is_ltx:
            return remaining.pop(0) if remaining else None
        if remaining:
            return remaining.pop(0)
        class_lower = str(node.get("class_type", "")).lower()
        wants_advanced = class_lower == "ltx2loraloaderadvanced"
        for index, selection in enumerate(remaining):
            if _ltx_lora_prefers_advanced_loader(selection[0]) == wants_advanced:
                return remaining.pop(index)
        if wants_advanced:
            current_lora = str((node.get("inputs") or {}).get("lora_name") or "")
            if not _ltx_lora_prefers_advanced_loader(current_lora):
                for index, selection in enumerate(remaining):
                    if not _ltx_lora_prefers_advanced_loader(selection[0]):
                        return remaining.pop(index)
            return None
        for index, selection in enumerate(remaining):
            if not _ltx_lora_prefers_advanced_loader(selection[0]):
                return remaining.pop(index)
        return None

    existing_nodes = [
        (str(node_id), node)
        for node_id, node in api.items()
        if isinstance(node, dict)
        and "lora" in str(node.get("class_type", "")).lower()
        and isinstance(node.get("inputs"), dict)
        and not protected_ltx_lora_node(node)
    ]
    remaining = list(selections)
    for node_id, node in existing_nodes:
        selection = pop_selection_for_node(remaining, node)
        inputs = node.setdefault("inputs", {})
        if not selection:
            if is_ltx and str(node.get("class_type", "")).lower() == "ltx2loraloaderadvanced":
                current_lora = str(inputs.get("lora_name") or "")
                if not _ltx_lora_prefers_advanced_loader(current_lora):
                    model_value = inputs.get("model")
                    if isinstance(model_value, list):
                        _replace_model_refs(api, [node_id, 0], list(model_value))
                    api.pop(node_id, None)
                    continue
            if "strength_model" in inputs:
                inputs["strength_model"] = 0.0
            elif "strength" in inputs:
                inputs["strength"] = 0.0
            if "strength_clip" in inputs:
                inputs["strength_clip"] = 0.0
            if is_ltx and str(node.get("class_type", "")).lower() == "ltx2loraloaderadvanced":
                patch_ltx2_lora_inputs(inputs, 0.0)
            continue
        lora_name, strength_model, strength_clip = selection
        if is_ltx and str(node.get("class_type", "")).lower() != "ltx2loraloaderadvanced":
            model_value = inputs.get("model")
            node["class_type"] = "LTX2LoraLoaderAdvanced"
            node.setdefault("_meta", {})["title"] = f"LTX LoRA - {Path(lora_name).name}"
            inputs.clear()
            if model_value is not None:
                inputs["model"] = model_value
        strength_model = effective_strength(lora_name, strength_model)
        if model_only_lora and str(node.get("class_type", "")).lower() != "ltx2loraloaderadvanced":
            model_value = inputs.get("model")
            node["class_type"] = "LoraLoaderModelOnly"
            inputs.clear()
            if model_value is not None:
                inputs["model"] = model_value
        elif not model_only_lora and str(node.get("class_type", "")).lower() == "loraloader":
            model_value = inputs.get("model")
            clip_value = inputs.get("clip")
            inputs.clear()
            if model_value is not None:
                inputs["model"] = model_value
            if clip_value is not None:
                inputs["clip"] = clip_value
        inputs["lora_name"] = lora_name
        if model_only_lora:
            inputs["strength_model"] = strength_model
            inputs.pop("strength", None)
        else:
            inputs["strength_model"] = strength_model
            inputs["strength_clip"] = strength_clip
        if "strength_clip" in inputs:
            inputs["strength_clip"] = strength_clip
        if is_ltx and str(node.get("class_type", "")).lower() == "ltx2loraloaderadvanced":
            patch_ltx2_lora_inputs(inputs, strength_model)
    if not remaining:
        return

    target_refs = _model_input_refs(api)
    if not target_refs:
        return
    original_model_ref = target_refs[0]
    clip_ref = None if model_only_lora else _find_clip_ref_for_lora(api, original_model_ref)
    model_ref = list(original_model_ref)
    final_clip_ref = list(clip_ref) if clip_ref else None
    next_id = _next_api_node_id(api)

    for lora_name, strength_model, strength_clip in remaining:
        strength_model = effective_strength(lora_name, strength_model)
        node_id = str(next_id)
        if final_clip_ref:
            api[node_id] = {
                "class_type": "LoraLoader",
                "inputs": {
                    "model": model_ref,
                    "clip": final_clip_ref,
                    "lora_name": lora_name,
                    "strength_model": strength_model,
                    "strength_clip": strength_clip,
                },
                "_meta": {"title": f"LoRA - {Path(lora_name).name}"},
            }
            model_ref = [node_id, 0]
            final_clip_ref = [node_id, 1]
        else:
            if is_ltx and _ltx_lora_prefers_advanced_loader(lora_name):
                routed_strength = max(0.0, min(1.0, abs(float(strength_model))))
                api[node_id] = {
                    "class_type": "LTX2LoraLoaderAdvanced",
                    "inputs": {
                        "model": model_ref,
                        "lora_name": lora_name,
                        "strength_model": strength_model,
                        "video": routed_strength,
                        "video_to_audio": ltx_audio_strength(strength_model),
                        "audio": ltx_audio_strength(strength_model),
                        "audio_to_video": ltx_audio_strength(strength_model),
                        "other": routed_strength,
                    },
                    "_meta": {"title": f"LTX LoRA - {Path(lora_name).name}"},
                }
            else:
                api[node_id] = {
                    "class_type": "LoraLoaderModelOnly",
                    "inputs": {
                        "model": model_ref,
                        "lora_name": lora_name,
                        "strength_model": strength_model,
                    },
                    "_meta": {"title": f"LoRA - {Path(lora_name).name}"},
                }
            model_ref = [node_id, 0]
        next_id += 1

    _replace_model_refs(api, original_model_ref, model_ref)
    if clip_ref and final_clip_ref:
        _replace_clip_refs(api, clip_ref, final_clip_ref)


def _model_input_refs(api: dict[str, Any]) -> list[list[Any]]:
    refs: list[list[Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if "lora" in class_lower:
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        value = inputs.get("model")
        if isinstance(value, list) and value:
            key = tuple(value)
            if key not in seen:
                refs.append(list(value))
                seen.add(key)
    return refs


def _find_clip_ref_for_lora(api: dict[str, Any], model_ref: list[Any]) -> list[Any] | None:
    if model_ref and str(model_ref[0]) in api:
        source = api.get(str(model_ref[0]))
        class_lower = str(source.get("class_type", "")).lower() if isinstance(source, dict) else ""
        if "checkpointloader" in class_lower:
            return [str(model_ref[0]), 1]
        if "loraloader" in class_lower:
            return [str(model_ref[0]), 1]
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if "cliploader" in class_lower or "dualcliploader" in class_lower:
            return [str(node_id), 0]
    for node in api.values():
        if not isinstance(node, dict):
            continue
        clip_input = node.get("inputs", {}).get("clip")
        if isinstance(clip_input, list) and clip_input:
            return list(clip_input)
    return None


def _replace_model_refs(api: dict[str, Any], old_ref: list[Any], new_ref: list[Any]) -> None:
    for node in api.values():
        if not isinstance(node, dict):
            continue
        if "lora" in str(node.get("class_type", "")).lower():
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and inputs.get("model") == old_ref:
            inputs["model"] = list(new_ref)


def _replace_clip_refs(api: dict[str, Any], old_ref: list[Any], new_ref: list[Any]) -> None:
    for node in api.values():
        if not isinstance(node, dict):
            continue
        if "lora" in str(node.get("class_type", "")).lower():
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and inputs.get("clip") == old_ref:
            inputs["clip"] = list(new_ref)


def _ensure_inpaint_mask_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, str]) -> None:
    if request.activity != "img2img" or not _uses_inpaint_mask_mode(request):
        return
    mask_image_name = assets.get("mask_image")
    if not mask_image_name:
        return
    if any(str(node.get("class_type", "")).lower() == "vaeencodeforinpaint" for node in api.values() if isinstance(node, dict)):
        return

    sampler_node_id = _find_sampler_node_id(api)
    if not sampler_node_id:
        return
    reference_node_id = _find_reference_image_node_id(api, assets.get("reference_image"))
    if not reference_node_id:
        reference_node_id = _add_load_image_node(api, assets.get("reference_image"), "Reference Image")
    vae_ref = _find_vae_ref(api)
    if not reference_node_id or not vae_ref:
        return

    _append_inpaint_mask(
        api,
        request,
        reference_node_id=reference_node_id,
        vae_ref=vae_ref,
        sampler_node_id=sampler_node_id,
        mask_image_name=mask_image_name,
        start_id=_next_api_node_id(api),
    )


def _ensure_controlnet_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, str]) -> None:
    controlnet_name = assets.get("controlnet_model")
    controlnet_image_name = assets.get("controlnet_image")
    if not _controlnet_can_apply(request, controlnet_name, controlnet_image_name):
        return
    if (
        assets.get("controlnet_category") == "model_patches"
        and request.preset.lower() in {"qwen", "zimageturbo", "zimage"}
    ):
        return

    loader_id = None
    for node_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "controlnetloader":
            loader_id = str(node_id)
            node.setdefault("inputs", {})["control_net_name"] = controlnet_name
            node.pop("bypassed", None)
            break

    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower not in {"controlnetapply", "controlnetapplyadvanced"}:
            continue
        inputs = node.setdefault("inputs", {})
        if loader_id:
            inputs["control_net"] = [loader_id, 0]
        control_image_id = _find_controlnet_image_node_id(api)
        if not control_image_id:
            control_image_id = _add_load_image_node(api, controlnet_image_name, "ControlNet Image")
        if control_image_id:
            inputs["image"] = [control_image_id, 0]
        inputs["strength"] = max(0.0, min(10.0, float(request.controlnet.strength or 0.75)))
        if "start_percent" in inputs:
            inputs["start_percent"] = max(0.0, min(1.0, float(request.controlnet.start_percent or 0.0)))
        if "end_percent" in inputs:
            inputs["end_percent"] = max(0.0, min(1.0, float(request.controlnet.end_percent or 1.0)))
        node.pop("bypassed", None)
        return

    sampler_node_id = _find_sampler_node_id(api)
    if not sampler_node_id:
        return
    sampler = api.get(sampler_node_id, {})
    sampler_inputs = sampler.get("inputs", {}) if isinstance(sampler, dict) else {}
    positive_ref = sampler_inputs.get("positive")
    negative_ref = sampler_inputs.get("negative")
    if not isinstance(positive_ref, list) or not isinstance(negative_ref, list):
        return
    positive_ref, negative_ref, _ = _append_controlnet_route(
        api,
        request,
        positive_ref=list(positive_ref),
        negative_ref=list(negative_ref),
        controlnet_name=controlnet_name,
        controlnet_image_name=controlnet_image_name,
        vae_ref=_find_vae_ref(api),
        start_id=_next_api_node_id(api),
    )
    sampler_inputs["positive"] = positive_ref
    sampler_inputs["negative"] = negative_ref


def _ensure_ltx_workflow_extensions(api: dict[str, Any], request: GenerateRequest, assets: dict[str, str]) -> None:
    if request.preset.lower() != "ltx":
        return
    _apply_ltx_tiled_decode_settings(api, request)
    _ensure_ltx_detailer_lora(api, request, assets)
    _ensure_ltx_ic_lora_control_route(api, request, assets)


def _bool_option(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() not in {"", "0", "false", "off", "none", "no"}


def _int_option(value: Any, default: int, minimum: int | None = None, maximum: int | None = None) -> int:
    try:
        numeric = int(round(float(value)))
    except (TypeError, ValueError):
        numeric = default
    if minimum is not None:
        numeric = max(minimum, numeric)
    if maximum is not None:
        numeric = min(maximum, numeric)
    return numeric


def _apply_ltx_tiled_decode_settings(api: dict[str, Any], request: GenerateRequest) -> None:
    video_options = request.video or {}
    tiles_x = _int_option(video_options.get("decode_tiles_x") or video_options.get("tiled_decode_x"), 2, 1, 8)
    tiles_y = _int_option(video_options.get("decode_tiles_y") or video_options.get("tiled_decode_y"), 2, 1, 8)
    overlap = _int_option(video_options.get("decode_overlap") or video_options.get("tiled_decode_overlap"), 6, 0, 256)
    working_device = str(video_options.get("decode_working_device") or "auto")
    working_dtype = str(video_options.get("decode_working_dtype") or "auto")
    for node in api.values():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "ltxvtiledvaedecode":
            continue
        inputs = node.setdefault("inputs", {})
        inputs["horizontal_tiles"] = tiles_x
        inputs["vertical_tiles"] = tiles_y
        inputs["overlap"] = overlap
        inputs["working_device"] = working_device
        inputs["working_dtype"] = working_dtype


def _ensure_ltx_detailer_lora(api: dict[str, Any], request: GenerateRequest, assets: dict[str, str]) -> None:
    video_options = request.video or {}
    if not _bool_option(video_options.get("detailer_enabled"), False):
        return
    detailer_name = _selected_text(video_options.get("detailer_lora")) or assets.get("detailer_lora")
    if not detailer_name:
        return
    strength = _number_or_none(video_options.get("detailer_strength"))
    strength_model = 1.0 if strength is None else max(-2.0, min(2.0, strength))
    for node in api.values():
        if not isinstance(node, dict):
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict) or "lora_name" not in inputs:
            continue
        if "detailer" in title or "detailer" in str(inputs.get("lora_name", "")).lower():
            inputs["lora_name"] = detailer_name
            inputs["strength_model"] = strength_model
            return

    target_model_inputs: list[tuple[dict[str, Any], list[Any]]] = []
    for node in api.values():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        title = str(node.get("_meta", {}).get("title", "")).lower()
        inputs = node.get("inputs", {})
        if (
            class_lower == "cfgguider"
            and isinstance(inputs, dict)
            and isinstance(inputs.get("model"), list)
            and any(token in title for token in ("refiner", "upscale", "detailer"))
        ):
            target_model_inputs.append((inputs, list(inputs["model"])))
    if not target_model_inputs:
        target_refs = _model_input_refs(api)
        if not target_refs:
            return
        original_model_ref = target_refs[0]
        node_id = str(_next_api_node_id(api))
        api[node_id] = {
            "class_type": "LoraLoaderModelOnly",
            "inputs": {
                "model": list(original_model_ref),
                "lora_name": detailer_name,
                "strength_model": strength_model,
            },
            "_meta": {"title": f"LTX Detailer LoRA - {Path(detailer_name).name}"},
        }
        _replace_model_refs(api, original_model_ref, [node_id, 0])
        return
    detailer_refs: dict[tuple[Any, ...], list[Any]] = {}
    for inputs, original_model_ref in target_model_inputs:
        key = tuple(original_model_ref)
        detailer_ref = detailer_refs.get(key)
        if detailer_ref is None:
            node_id = str(_next_api_node_id(api))
            api[node_id] = {
                "class_type": "LoraLoaderModelOnly",
                "inputs": {
                    "model": list(original_model_ref),
                    "lora_name": detailer_name,
                    "strength_model": strength_model,
                },
                "_meta": {"title": f"LTX Refiner Detailer LoRA - {Path(detailer_name).name}"},
            }
            detailer_ref = [node_id, 0]
            detailer_refs[key] = detailer_ref
        inputs["model"] = list(detailer_ref)


def _ensure_ltx_ic_lora_control_route(api: dict[str, Any], request: GenerateRequest, assets: dict[str, str]) -> None:
    control = request.controlnet
    if not control.enabled:
        return
    control_image_name = assets.get("controlnet_image")
    ic_lora_name = assets.get("controlnet_model") or assets.get("ic_lora")
    if not control_image_name or not ic_lora_name:
        return

    control_image_id = _find_controlnet_image_node_id(api) or _add_load_image_node(api, control_image_name, "LTX IC-LoRA Control Image")
    if not control_image_id:
        return
    api[str(control_image_id)].setdefault("inputs", {})["image"] = control_image_name
    api[str(control_image_id)].setdefault("_meta", {})["title"] = "LTX IC-LoRA Control Image"

    ic_loader_id = _ensure_ltx_ic_lora_loader(api, ic_lora_name, max(-2.0, min(2.0, float(control.strength or 0.75))))
    if not ic_loader_id:
        return

    guide_inputs = _ltx_ic_lora_guide_inputs(api)
    if not guide_inputs:
        return
    positive_ref, negative_ref, latent_ref, vae_ref, rewires = guide_inputs
    if not positive_ref or not negative_ref or not latent_ref or not vae_ref:
        return

    has_director_guide = any(
        isinstance(node, dict) and str(node.get("class_type", "")).lower() == "ltxdirectorguide"
        for node in api.values()
    )
    if has_director_guide and not any(token in str(ic_lora_name).lower() for token in ("union", "control", "ref")):
        return
    existing_guide_id = None
    for node_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "ltxaddvideoicloraguide":
            existing_guide_id = str(node_id)
            break
    ic_guide_id = existing_guide_id or str(_next_api_node_id(api))
    frame_count = _number_or_none((request.video or {}).get("frames")) or _number_or_none((request.director or {}).get("duration_frames")) or 0
    frame_idx = _int_option(float(control.start_percent or 0.0) * max(0, frame_count), 0, 0, 9999)
    video_options = request.video or {}
    crop = str(video_options.get("ltx_ic_crop") or ("center" if str(control.balance).lower().startswith("control") else "disabled"))
    if crop not in {"disabled", "center"}:
        crop = "disabled"
    guide_node = {
        "class_type": "LTXAddVideoICLoRAGuide",
        "inputs": {
            "positive": list(positive_ref),
            "negative": list(negative_ref),
            "vae": list(vae_ref),
            "latent": list(latent_ref),
            "image": [str(control_image_id), 0],
            "frame_idx": frame_idx,
            "strength": max(0.0, min(1.0, float(control.strength or 0.8))),
            "latent_downscale_factor": 1.0 if has_director_guide else [str(ic_loader_id), 1],
            "crop": crop,
            "use_tiled_encode": _bool_option(video_options.get("ltx_ic_tiled_encode"), False),
            "tile_size": _int_option(video_options.get("ltx_ic_tile_size"), 256, 64, 512),
            "tile_overlap": _int_option(video_options.get("ltx_ic_tile_overlap"), 64, 16, 256),
        },
        "_meta": {"title": "LTX IC-LoRA Control Guide"},
    }
    if existing_guide_id:
        api[existing_guide_id].setdefault("inputs", {}).update(guide_node["inputs"])
        api[existing_guide_id]["_meta"] = guide_node["_meta"]
    else:
        api[ic_guide_id] = guide_node

    for owner_id, key, slot in rewires:
        if str(owner_id) == str(ic_guide_id):
            continue
        owner = api.get(str(owner_id))
        if not isinstance(owner, dict):
            continue
        inputs = owner.setdefault("inputs", {})
        if not isinstance(inputs, dict):
            continue
        inputs[key] = [str(ic_guide_id), slot]


def _ensure_ltx_ic_lora_loader(api: dict[str, Any], lora_name: str, strength_model: float) -> str | None:
    existing_id = None
    for node_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "ltxicloraloadermodelonly":
            existing_id = str(node_id)
            inputs = node.setdefault("inputs", {})
            inputs["lora_name"] = lora_name
            inputs["strength_model"] = strength_model
            break

    model_ref: list[Any] | None = None
    director_id = None
    for node_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "ltxdirector":
            inputs = node.setdefault("inputs", {})
            if isinstance(inputs.get("model"), list):
                model_ref = list(inputs["model"])
                director_id = str(node_id)
                break
    if model_ref is None:
        for _node_id, node in api.items():
            if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "cfgguider":
                inputs = node.setdefault("inputs", {})
                if isinstance(inputs.get("model"), list):
                    model_ref = list(inputs["model"])
                    break
    if model_ref is None:
        refs = _model_input_refs(api)
        model_ref = refs[0] if refs else None
    if model_ref is None:
        return existing_id

    if existing_id:
        inputs = api[existing_id].setdefault("inputs", {})
        inputs.setdefault("model", list(model_ref))
        loader_ref = [existing_id, 0]
    else:
        existing_id = str(_next_api_node_id(api))
        api[existing_id] = {
            "class_type": "LTXICLoRALoaderModelOnly",
            "inputs": {
                "model": list(model_ref),
                "lora_name": lora_name,
                "strength_model": strength_model,
            },
            "_meta": {"title": f"LTX IC-LoRA Control - {Path(lora_name).name}"},
        }
        loader_ref = [existing_id, 0]
        _replace_model_refs(api, model_ref, loader_ref)
    if director_id:
        api[director_id].setdefault("inputs", {})["model"] = loader_ref
    return existing_id


def _ltx_ic_lora_guide_inputs(api: dict[str, Any]) -> tuple[list[Any], list[Any], list[Any], list[Any], list[tuple[str, str, int]]] | None:
    for node_id, node in api.items():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "ltxdirectorguide":
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        positive_ref = inputs.get("positive")
        negative_ref = inputs.get("negative")
        latent_ref = inputs.get("latent")
        if not isinstance(positive_ref, list) or not isinstance(negative_ref, list) or not isinstance(latent_ref, list):
            continue
        rewires = [(str(node_id), "positive", 0), (str(node_id), "negative", 1), (str(node_id), "latent", 2)]
        return list(positive_ref), list(negative_ref), list(latent_ref), list(inputs.get("vae") or _find_ltx_video_vae_ref(api) or []), rewires

    cfg_id = None
    cfg_inputs: dict[str, Any] | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "cfgguider":
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and isinstance(inputs.get("positive"), list) and isinstance(inputs.get("negative"), list):
            cfg_id = str(node_id)
            cfg_inputs = inputs
            break
    if not cfg_id or not cfg_inputs:
        return None

    latent_owner_id = None
    latent_owner_key = None
    latent_ref = None
    for node_id, node in api.items():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "ltxvconcatavlatent":
            continue
        inputs = node.get("inputs", {})
        if isinstance(inputs, dict) and isinstance(inputs.get("video_latent"), list):
            latent_owner_id = str(node_id)
            latent_owner_key = "video_latent"
            latent_ref = list(inputs["video_latent"])
            break
    if latent_ref is None:
        sampler_id = _find_sampler_node_id(api)
        sampler = api.get(sampler_id or "")
        sampler_inputs = sampler.get("inputs", {}) if isinstance(sampler, dict) else {}
        if isinstance(sampler_inputs, dict) and isinstance(sampler_inputs.get("latent_image"), list):
            latent_owner_id = str(sampler_id)
            latent_owner_key = "latent_image"
            latent_ref = list(sampler_inputs["latent_image"])
    if latent_ref is None or not latent_owner_id or not latent_owner_key:
        return None

    positive_ref = list(cfg_inputs["positive"])
    negative_ref = list(cfg_inputs["negative"])
    rewires = _matching_input_refs(api, positive_ref, 0) + _matching_input_refs(api, negative_ref, 1)
    rewires.append((latent_owner_id, latent_owner_key, 2))
    vae_ref = _find_ltx_video_vae_ref(api) or _find_vae_ref(api)
    if not vae_ref:
        return None
    return positive_ref, negative_ref, latent_ref, list(vae_ref), rewires


def _matching_input_refs(api: dict[str, Any], target_ref: list[Any], output_slot: int) -> list[tuple[str, str, int]]:
    matches: list[tuple[str, str, int]] = []
    target = list(target_ref)
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        for key, value in inputs.items():
            if isinstance(value, list) and list(value) == target:
                matches.append((str(node_id), key, output_slot))
    return matches


def _find_ltx_video_vae_ref(api: dict[str, Any]) -> list[Any] | None:
    for node in api.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        if not isinstance(inputs, dict):
            continue
        vae_input = inputs.get("vae")
        if isinstance(vae_input, list) and ("ltxdirectorguide" in str(node.get("class_type", "")).lower() or "ltxvimgtovideo" in str(node.get("class_type", "")).lower()):
            return list(vae_input)
    for node_id, node in api.items():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "vaeloader":
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        vae_name = str(node.get("inputs", {}).get("vae_name", "")).lower()
        if "video" in title or ("ltx" in vae_name and "audio" not in vae_name and "tae" not in vae_name and "preview" not in vae_name):
            return [str(node_id), 0]
    return _find_vae_ref(api)


def _find_controlnet_image_node_id(api: dict[str, Any]) -> str | None:
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "loadimage":
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        if "control" in title:
            return str(node_id)
    return None


def _next_api_node_id(api: dict[str, Any]) -> int:
    numeric_ids = [int(key) for key in api.keys() if str(key).isdigit()]
    return max(numeric_ids, default=79) + 1


def _find_qwen_model_loader_ref(api: dict[str, Any]) -> list[Any] | None:
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower not in {"unetloader", "unetloadergguf"}:
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        inputs = node.get("inputs", {})
        haystack = " ".join([title, str(inputs.get("unet_name", "")) if isinstance(inputs, dict) else ""])
        if "qwen" in haystack.lower():
            return [str(node_id), 0]
    return None


def _find_sampler_node_id(api: dict[str, Any]) -> str | None:
    for node_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "ksampler":
            return str(node_id)
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        class_lower = str(node.get("class_type", "")).lower()
        if isinstance(inputs, dict) and "latent_image" in inputs and "sampler" in class_lower:
            return str(node_id)
    return None


def _ensure_conditioning_method_node(api: dict[str, Any], title: str, conditioning_ref: list[Any]) -> str:
    wanted = title.lower()
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        if str(node.get("class_type", "")).lower() != "fluxkontextmultireferencelatentmethod":
            continue
        node_title = str(node.get("_meta", {}).get("title", "")).lower()
        if node_title == wanted:
            inputs = node.setdefault("inputs", {})
            if isinstance(inputs, dict):
                inputs["conditioning"] = conditioning_ref
                inputs["reference_latents_method"] = "index_timestep_zero"
            return str(node_id)
    node_id = str(_next_api_node_id(api))
    api[node_id] = {
        "class_type": "FluxKontextMultiReferenceLatentMethod",
        "inputs": {"conditioning": conditioning_ref, "reference_latents_method": "index_timestep_zero"},
        "_meta": {"title": title},
    }
    return node_id


def _find_reference_image_node_id(api: dict[str, Any], reference_image_name: str | None) -> str | None:
    reference_lower = str(reference_image_name or "").lower()
    fallback: str | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if class_lower != "loadimage":
            continue
        title = str(node.get("_meta", {}).get("title", "")).lower()
        image_name = str(node.get("inputs", {}).get("image", "")).lower()
        if "mask" in title:
            continue
        if reference_lower and image_name == reference_lower:
            return str(node_id)
        if "reference" in title or "input" in title or "image" in title:
            fallback = fallback or str(node_id)
    return fallback


def _find_qwen_reference_loader_id(api: dict[str, Any], index: int, reference_image_name: str | None) -> str | None:
    reference_lower = str(reference_image_name or "").lower()
    target_titles = {
        f"reference image {index}",
        f"picture {index}",
        f"image {index}",
    }
    fallback: str | None = None
    for node_id, node in api.items():
        if not isinstance(node, dict) or str(node.get("class_type", "")).lower() != "loadimage":
            continue
        inputs = node.get("inputs", {})
        title = str(node.get("_meta", {}).get("title", "")).strip().lower()
        image_name = str(inputs.get("image", "")).lower() if isinstance(inputs, dict) else ""
        if reference_lower and image_name == reference_lower:
            return str(node_id)
        if title in target_titles:
            fallback = fallback or str(node_id)
    if index == 1:
        return fallback or _find_reference_image_node_id(api, reference_image_name)
    return fallback


def _add_load_image_node(api: dict[str, Any], image_name: str | None, title: str) -> str | None:
    if not image_name:
        return None
    node_id = str(_next_api_node_id(api))
    api[node_id] = {
        "class_type": "LoadImage",
        "inputs": {"image": image_name},
        "_meta": {"title": title},
    }
    return node_id


def _find_vae_ref(api: dict[str, Any]) -> list[Any] | None:
    for node_id, node in api.items():
        if isinstance(node, dict) and str(node.get("class_type", "")).lower() == "vaeloader":
            return [str(node_id), 0]
    for node in api.values():
        if not isinstance(node, dict):
            continue
        vae_input = node.get("inputs", {}).get("vae")
        if isinstance(vae_input, list) and vae_input:
            return vae_input
    for node_id, node in api.items():
        if not isinstance(node, dict):
            continue
        class_lower = str(node.get("class_type", "")).lower()
        if "checkpointloader" in class_lower:
            return [str(node_id), 2]
    return None


def _number_or_none(value: Any) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _looks_like_model_file(value: str) -> bool:
    lower = value.lower().strip()
    return lower.endswith((".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf", ".onnx"))


def _replacement_for_model_input(
    key: str,
    value: str,
    class_type: str,
    title: str,
    assets: dict[str, str],
    lora_slot: int,
) -> tuple[str | None, int]:
    haystack = " ".join([key, value, class_type, title]).lower()
    checkpoint_keys = {"ckpt_name", "unet_name", "model_name", "checkpoint_name"}
    if "upscale" in haystack and assets.get("latent_upscale"):
        return assets["latent_upscale"], lora_slot
    if key in checkpoint_keys and "projection" not in haystack and "proj" not in haystack:
        if assets.get("primary_model"):
            return assets["primary_model"], lora_slot
    if "high" in haystack and "wan" in haystack and assets.get("wan_high_model"):
        return assets["wan_high_model"], lora_slot
    if "low" in haystack and "wan" in haystack and assets.get("wan_low_model"):
        return assets["wan_low_model"], lora_slot
    if ("clip_l" in haystack or key == "clip_name1") and assets.get("flux_clip_l"):
        return assets["flux_clip_l"], lora_slot
    if "audio" in haystack and "vae" in haystack:
        if assets.get("audio_vae"):
            return assets["audio_vae"], lora_slot
        return None, lora_slot
    if ("tae" in haystack or "preview" in haystack) and assets.get("preview_vae"):
        return assets["preview_vae"], lora_slot
    if "vae" in haystack and assets.get("video_vae"):
        return assets["video_vae"], lora_slot
    if "vae" in haystack and assets.get("vae"):
        return assets["vae"], lora_slot
    if key == "text_encoder" and assets.get("text_encoder"):
        return assets["text_encoder"], lora_slot
    if ("projection" in haystack or "proj" in haystack) and assets.get("text_projection"):
        return assets["text_projection"], lora_slot
    if ("clip" in haystack or "text" in haystack or "gemma" in haystack) and assets.get("text_encoder"):
        return assets["text_encoder"], lora_slot
    if "lora" in haystack:
        lora_slot += 1
        if "outpaint" in haystack and assets.get("outpaint_lora"):
            return assets["outpaint_lora"], lora_slot
        if "ic" in haystack and assets.get("ic_lora"):
            return assets["ic_lora"], lora_slot
        if lora_slot == 1 and assets.get("distilled_lora_1"):
            return assets["distilled_lora_1"], lora_slot
        if assets.get("distilled_lora_2"):
            return assets["distilled_lora_2"], lora_slot
    if any(token in haystack for token in ["unet", "model", "checkpoint", "gguf", "diffusion"]):
        if assets.get("primary_model"):
            return assets["primary_model"], lora_slot
    return None, lora_slot
