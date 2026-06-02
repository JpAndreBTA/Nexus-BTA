from __future__ import annotations

import json
from importlib import metadata
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from packaging.version import Version

from .config import NexusSettings, runtime_python
from .scanner import scan_custom_nodes


_OPTIONAL_REQUIREMENT_FALLBACKS = {
    "onnxruntime-gpu": "onnxruntime>=1.18",
    "imath": "OpenEXR>=3.2.0",
}

_REQUIREMENT_VERSION_FLOORS = {
    "aiohttp": "aiohttp>=3.14.0",
    "av": "av>=17.0.1",
    "diffusers": "diffusers>=0.38.0",
    "fastapi": "fastapi>=0.136.3",
    "huggingface-hub": "huggingface-hub>=0.36.2",
    "kornia": "kornia>=0.8.2,<0.8.3",
    "numpy": "numpy>=2.4.6",
    "opencv-contrib-python": "opencv-contrib-python>=4.10.0.84",
    "opencv-python": "opencv-python>=4.10.0.84",
    "pillow": "pillow>=12.2.0",
    "protobuf": "protobuf>=5.29.6",
    "scikit-image": "scikit-image>=0.26.0",
    "tokenizers": "tokenizers>=0.22.2",
    "torch": "torch==2.10.0+cu130",
    "torchaudio": "torchaudio==2.10.0+cu130",
    "torchvision": "torchvision==0.25.0+cu130",
    "transformers": "transformers>=4.57.6,<5",
}

_OPTIONAL_REQUIREMENT_SKIPS = {
    "inference-cli",
    "inference-gpu",
}


def custom_node_requirements(settings: NexusSettings) -> dict[str, Path]:
    requirements: dict[str, Path] = {}
    for node in scan_custom_nodes(settings, include_references=False):
        path = Path(node.path) / "requirements.txt"
        if node.enabled and path.exists():
            requirements[node.name] = path
    return requirements


def _requirement_package_name(line: str) -> str | None:
    text = line.strip()
    if not text or text.startswith(("#", "-", "--")):
        return None
    egg_match = re.search(r"[#&]egg=([A-Za-z0-9_.-]+)", text)
    if egg_match:
        return egg_match.group(1)
    if text.lower().startswith(("git+", "http://", "https://")):
        return None
    match = re.match(r"([A-Za-z0-9_.-]+)", text)
    return match.group(1) if match else None


def _normalize_package_name(value: str) -> str:
    return value.lower().replace("_", "-")


def _installed_requirement(package: str) -> bool:
    normalized = _normalize_package_name(package)
    if normalized in _OPTIONAL_REQUIREMENT_SKIPS:
        return True
    floor = _REQUIREMENT_VERSION_FLOORS.get(normalized)
    for candidate in dict.fromkeys((package, normalized)):
        try:
            installed_version = metadata.version(candidate)
            if floor:
                floor_package = _requirement_package_name(floor)
                if floor_package and _normalize_package_name(floor_package) == normalized:
                    exact_match = re.search(r"==\s*([A-Za-z0-9_.!+-]+)", floor)
                    floor_match = re.search(r">=\s*([A-Za-z0-9_.!+-]+)", floor)
                    ceiling_match = re.search(r"<\s*([A-Za-z0-9_.!+-]+)", floor)
                    if exact_match and Version(installed_version) != Version(exact_match.group(1)):
                        return False
                    if floor_match and Version(installed_version) < Version(floor_match.group(1)):
                        return False
                    if ceiling_match and Version(installed_version) >= Version(ceiling_match.group(1)):
                        return False
            return True
        except metadata.PackageNotFoundError:
            pass
    fallback = _OPTIONAL_REQUIREMENT_FALLBACKS.get(normalized)
    if not fallback:
        return False
    fallback_package = _requirement_package_name(fallback)
    if not fallback_package:
        return False
    try:
        metadata.version(fallback_package)
        return True
    except metadata.PackageNotFoundError:
        return False


def _sanitized_requirements_file(requirements_path: Path, temp_dir: Path) -> Path:
    lines = requirements_path.read_text(encoding="utf-8", errors="ignore").splitlines()
    changed = False
    sanitized: list[str] = []
    for line in lines:
        package = _requirement_package_name(line)
        normalized = _normalize_package_name(package or "")
        fallback = _OPTIONAL_REQUIREMENT_FALLBACKS.get(normalized)
        floor = _REQUIREMENT_VERSION_FLOORS.get(normalized)
        if fallback:
            sanitized.append(f"{fallback}  # Nexus fallback for optional {package}")
            changed = True
        elif floor:
            sanitized.append(f"{floor}  # Nexus minimum for current Comfy/Nexus runtime")
            changed = True
        elif normalized in _OPTIONAL_REQUIREMENT_SKIPS:
            sanitized.append(f"# {line}  # Nexus skip: optional Roboflow inference package conflicts with current Comfy/Nexus runtime")
            changed = True
        else:
            sanitized.append(line)
    if not changed:
        return requirements_path
    target = temp_dir / f"{requirements_path.parent.name}-requirements.txt"
    target.write_text("\n".join(sanitized) + "\n", encoding="utf-8")
    return target


def custom_node_dependency_status(settings: NexusSettings) -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name, path in custom_node_requirements(settings).items():
        missing: list[str] = []
        packages: list[str] = []
        try:
            lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
        except Exception as exc:
            status[name] = {"path": str(path), "installed": False, "packages": [], "missing": [str(exc)]}
            continue
        for line in lines:
            package = _requirement_package_name(line)
            if not package:
                continue
            if _normalize_package_name(package) in _OPTIONAL_REQUIREMENT_SKIPS:
                continue
            packages.append(package)
            if not _installed_requirement(package):
                missing.append(package)
        status[name] = {
            "path": str(path),
            "installed": not missing,
            "packages": packages,
            "missing": missing,
        }
    return status


def install_custom_node_dependencies(
    settings: NexusSettings,
    node_names: list[str] | None = None,
    all_enabled: bool = False,
) -> tuple[list[str], dict[str, str]]:
    requested_targets = list(node_names or [])
    cloned: list[str] = []
    clone_errors: dict[str, str] = {}
    for target in requested_targets:
        if not str(target).lower().startswith(("http://", "https://", "git@")):
            continue
        name, error = _clone_custom_node_repo(settings, str(target))
        if error:
            clone_errors[str(target)] = error
        elif name:
            cloned.append(name)

    requirements = custom_node_requirements(settings)
    selected: dict[str, Path]
    if all_enabled:
        selected = requirements
    else:
        wanted = {name.lower() for name in requested_targets if not name.lower().startswith(("http://", "https://", "git@"))}
        wanted.update(name.lower() for name in cloned)
        selected = {name: path for name, path in requirements.items() if name.lower() in wanted}

    installed: list[str] = cloned[:]
    errors: dict[str, str] = dict(clone_errors)
    python_exe = runtime_python(settings)
    with tempfile.TemporaryDirectory(prefix="nexus-node-deps-") as temp_root:
        temp_dir = Path(temp_root)
        for name, requirements_path in selected.items():
            try:
                install_path = _sanitized_requirements_file(requirements_path, temp_dir)
                result = subprocess.run(
                    [str(python_exe), "-m", "pip", "install", "-r", str(install_path)],
                    cwd=str(settings.project_root),
                    capture_output=True,
                    text=True,
                    timeout=900,
                )
                if result.returncode == 0:
                    installed.append(name)
                else:
                    errors[name] = (result.stderr or result.stdout or "pip install failed").strip()[-2000:]
            except Exception as exc:
                errors[name] = str(exc)
    return installed, errors


def _clone_custom_node_repo(settings: NexusSettings, repo: str) -> tuple[str | None, str | None]:
    settings.custom_nodes_dir.mkdir(parents=True, exist_ok=True)
    name = _safe_repo_dir_name(repo)
    if not name:
        return None, f"Invalid repository target: {repo}"
    target = settings.custom_nodes_dir / name
    if target.exists():
        return name, None
    try:
        result = subprocess.run(
            ["git", "clone", "--depth", "1", repo, str(target)],
            cwd=str(settings.custom_nodes_dir),
            capture_output=True,
            text=True,
            timeout=900,
        )
        if result.returncode != 0:
            return None, (result.stderr or result.stdout or "git clone failed").strip()[-2000:]
        return name, None
    except Exception as exc:
        return None, str(exc)


def manager_suggestions_for_nodes(settings: NexusSettings, missing_nodes: list[str]) -> dict[str, list[str]]:
    if not missing_nodes:
        return {}

    cache_dir = settings.user_dir / "__manager" / "cache"
    maps = sorted(cache_dir.glob("*extension-node-map*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not maps:
        return {}

    missing = {node.lower(): node for node in missing_nodes}
    suggestions: dict[str, list[str]] = {node: [] for node in missing_nodes}
    for node in missing_nodes:
        canonical = _canonical_repo_for_missing_node(node)
        if canonical:
            suggestions.setdefault(node, []).append(canonical)
    try:
        data: dict[str, Any] = json.loads(maps[0].read_text(encoding="utf-8"))
    except Exception:
        return {}

    for repo, value in data.items():
        node_names: list[str] = []
        if isinstance(value, list) and value:
            if isinstance(value[0], list):
                node_names = [str(item) for item in value[0]]
            elif isinstance(value[0], str):
                node_names = [str(item) for item in value]
        for node_name in node_names:
            key = node_name.lower()
            if key in missing:
                suggestions.setdefault(missing[key], []).append(repo)

    return {node: list(dict.fromkeys(repos))[:5] for node, repos in suggestions.items() if repos}


def custom_nodes_for_workflow(
    settings: NexusSettings,
    class_types: list[str],
    suggestions: dict[str, list[str]] | None = None,
) -> list[str]:
    installed_nodes = [node for node in scan_custom_nodes(settings, include_references=False) if node.enabled]
    installed_by_key = {
        _normalize_name(node.name): node.name
        for node in installed_nodes
        if _normalize_name(node.name)
    }
    for node in installed_nodes:
        path_key = _normalize_name(Path(node.path).name)
        if path_key:
            installed_by_key.setdefault(path_key, node.name)

    targets: set[str] = set()
    workflow_classes = {item.lower(): item for item in class_types}
    for class_type in class_types:
        heuristic = _heuristic_custom_node_name(class_type)
        if heuristic and _normalize_name(heuristic) in installed_by_key:
            targets.add(installed_by_key[_normalize_name(heuristic)])

    for repo in _repos_for_classes(settings, workflow_classes):
        match = _match_repo_to_installed_node(repo, installed_by_key)
        if match:
            targets.add(match)

    for repos in (suggestions or {}).values():
        fallback_repo = ""
        for repo in repos:
            match = _match_repo_to_installed_node(repo, installed_by_key)
            if match:
                targets.add(match)
                fallback_repo = ""
                break
            if not fallback_repo and repo.lower().startswith(("http://", "https://", "git@")):
                fallback_repo = repo
        if fallback_repo:
            targets.add(fallback_repo)

    requirements = custom_node_requirements(settings)
    return sorted(
        name
        for name in targets
        if name in requirements or name.lower().startswith(("http://", "https://", "git@"))
    )


def _repos_for_classes(settings: NexusSettings, class_types: dict[str, str]) -> set[str]:
    cache_dir = settings.user_dir / "__manager" / "cache"
    maps = sorted(cache_dir.glob("*extension-node-map*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not maps or not class_types:
        return set()
    try:
        data: dict[str, Any] = json.loads(maps[0].read_text(encoding="utf-8"))
    except Exception:
        return set()

    repos: set[str] = set()
    wanted = set(class_types.keys())
    normalized_wanted = {_normalize_node_class(name): name for name in class_types}
    for repo, value in data.items():
        node_names = _node_names_from_manager_value(value)
        for node_name in node_names:
            normalized = _normalize_node_class(node_name)
            if node_name.lower() in wanted or normalized in normalized_wanted:
                repos.add(repo)
                break
    return repos


def _node_names_from_manager_value(value: Any) -> list[str]:
    if isinstance(value, list) and value:
        if isinstance(value[0], list):
            return [str(item) for item in value[0]]
        if isinstance(value[0], str):
            return [str(item) for item in value]
    return []


def _match_repo_to_installed_node(repo: str, installed_by_key: dict[str, str]) -> str | None:
    repo_key = _normalize_name(repo.rstrip("/").removesuffix(".git").split("/")[-1])
    if repo_key in installed_by_key:
        return installed_by_key[repo_key]
    for key, name in installed_by_key.items():
        if repo_key and (repo_key in key or key in repo_key):
            return name
    return None


def _heuristic_custom_node_name(class_type: str) -> str | None:
    value = class_type.lower()
    if value in {
        "ltxvaddguidemulti",
        "ltxvaddguidesfrombatch",
        "ltxvimgtovideoinplacekj",
        "ltx2_nag",
        "ltx2samplingpreviewoverride",
    }:
        return "ComfyUI-KJNodes"
    if "rgthree" in value or "fast groups bypasser" in value:
        return "rgthree-comfy"
    if "res4lyf" in value or "clown" in value or "bong" in value:
        return "RES4LYF"
    if "inttofloat" in value or "floattoint" in value or "int-and-float" in value:
        return "comfyui-int-and-float"
    if "ultralyticsdetectorprovider" in value or "bbox_detector" in value or "samdetector" in value:
        return "ComfyUI-Impact-Subpack"
    if value.startswith("easy ") or "easy getnode" in value or "easy setnode" in value:
        return "comfyui-easy-use"
    if "melbandroformer" in value or "roformer" in value:
        return "ComfyUI-MelBandRoFormer"
    if "gguf" in value:
        return "ComfyUI-GGUF"
    if "ltx" in value:
        return "ComfyUI-LTXVideo"
    if "video combine" in value or "vhs_" in value:
        return "comfyui-videohelpersuite"
    if "lanpaint" in value:
        return "LanPaint"
    if "qwenmultiangle" in value:
        return "ComfyUI-qwenmultiangle"
    return None


def _canonical_repo_for_missing_node(class_type: str) -> str | None:
    value = class_type.lower()
    if value in {
        "ltxvaddguidemulti",
        "ltxvaddguidesfrombatch",
        "ltxvimgtovideoinplacekj",
        "ltx2_nag",
        "ltx2samplingpreviewoverride",
    }:
        return "https://github.com/kijai/ComfyUI-KJNodes"
    if "ultralyticsdetectorprovider" in value:
        return "https://github.com/ltdrdata/ComfyUI-Impact-Subpack"
    if "fast groups bypasser" in value or "rgthree" in value:
        return "https://github.com/rgthree/rgthree-comfy"
    if "res4lyf" in value or "clown" in value or "bong" in value:
        return "https://github.com/ClownsharkBatwing/RES4LYF"
    if "inttofloat" in value or "floattoint" in value or "int-and-float" in value:
        return "https://github.com/danTheMonk/comfyui-int-and-float"
    if value.startswith("easy ") or "easy getnode" in value or "easy setnode" in value:
        return "https://github.com/yolain/ComfyUI-Easy-Use"
    if "melbandroformer" in value or "roformer" in value:
        return "https://github.com/kijai/ComfyUI-MelBandRoFormer"
    if "ltx" in value:
        return "https://github.com/Lightricks/ComfyUI-LTXVideo"
    if "vhs_" in value or "loadvideo" in value or "getvideocomponents" in value or "createvideo" in value or "savevideo" in value or "videocombine" in value:
        return "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite"
    if "dwpreprocessor" in value or "openposepreprocessor" in value or "cannyedgepreprocessor" in value:
        return "https://github.com/Fannovel16/comfyui_controlnet_aux"
    if "seedvr2" in value:
        return "https://github.com/numz/seedvr2_videoupscaler"
    if "flashvsr" in value:
        return "https://github.com/ClownsharkBatwing/ComfyUI-FlashVSR"
    if "rtxvideosuperresolution" in value:
        return "https://github.com/ClownsharkBatwing/comfyui-rtx-simple"
    if value.startswith("pid") or "pidprepare" in value or "pidsample" in value or "pidfinalize" in value:
        return "https://github.com/Merserk/ComfyUI-PiD"
    if "lanpaint" in value:
        return "https://github.com/scraed/LanPaint"
    if "qwenmultiangle" in value:
        return "https://github.com/jtydhr88/ComfyUI-qwenmultiangle"
    return None


def _safe_repo_dir_name(repo: str) -> str:
    tail = repo.rstrip("/").removesuffix(".git").split("/")[-1]
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tail).strip("._-")


def _normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _normalize_node_class(value: str) -> str:
    return _normalize_name(value.replace("(rgthree)", "rgthree"))
