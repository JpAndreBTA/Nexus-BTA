from __future__ import annotations

import os
import shutil
from pathlib import Path

from .config import NexusSettings
from .schemas import ImportRequest


TARGETS = {
    "models": "models_dir",
    "custom_nodes": "custom_nodes_dir",
    "workflows": "workflows_dir",
    "comfy_core": "comfy_root",
    "python_env": "comfy_python",
}


def import_resource(settings: NexusSettings, request: ImportRequest) -> dict[str, str]:
    source = request.source
    if not source.exists():
        raise FileNotFoundError(str(source))

    if request.target_kind == "python_env":
        target_root = settings.comfy_python.parent.parent
    else:
        target_root = getattr(settings, TARGETS[request.target_kind])

    if request.target_kind in {"comfy_core", "python_env"}:
        target = target_root
    elif source.is_dir():
        target = target_root / source.name
    else:
        target = target_root / source.name

    if target.exists() and not request.overwrite:
        return {"status": "exists", "target": str(target)}
    if target.exists() and request.overwrite:
        if target.is_dir() and not target.is_symlink():
            shutil.rmtree(target)
        else:
            target.unlink()

    target.parent.mkdir(parents=True, exist_ok=True)
    if request.mode == "link":
        _link(source, target)
        status = "linked"
    else:
        _copy(source, target)
        status = "copied"
    return {"status": status, "target": str(target)}


def _copy(source: Path, target: Path) -> None:
    if source.is_dir():
        shutil.copytree(source, target, dirs_exist_ok=True)
    else:
        shutil.copy2(source, target)


def _link(source: Path, target: Path) -> None:
    if os.name == "nt" and source.is_dir():
        try:
            os.symlink(source, target, target_is_directory=True)
        except OSError:
            import subprocess

            subprocess.check_call(["cmd", "/c", "mklink", "/J", str(target), str(source)])
    else:
        os.symlink(source, target, target_is_directory=source.is_dir())

