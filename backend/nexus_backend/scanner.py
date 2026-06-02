from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from .config import NexusSettings
from .schemas import CustomNodeSummary, ModelCatalog, ModelFile


MODEL_CATEGORIES = [
    "checkpoints",
    "diffusion_models",
    "unet",
    "loras",
    "vae",
    "text_encoders",
    "clip",
    "clip_vision",
    "controlnet",
    "model_patches",
    "upscale_models",
    "latent_upscale_models",
    "video_restore_models",
    "denoise_models",
    "face_restore_models",
    "background_removal",
    "embeddings",
    "animatediff_models",
    "animatediff_motion_lora",
    "frame_interpolation",
    "style_models",
    "diffusers",
]

CHECKPOINT_PRESET_FOLDERS = [
    "sd15",
    "sdxl",
    "illustrious",
    "noobai",
    "animagine",
    "pony",
    "flux",
    "qwen",
    "lumina",
    "wan",
    "ltx",
    "anima",
    "zimage",
]

MODEL_SOURCE_CATEGORY_ALIASES = {
    "checkpoint": "checkpoints",
    "checkpoints": "checkpoints",
    "vae": "vae",
    "lora": "loras",
    "loras": "loras",
    "controlnet": "controlnet",
    "controlnets": "controlnet",
    "model_patch": "model_patches",
    "model_patches": "model_patches",
    "upscale": "upscale_models",
    "upscale_models": "upscale_models",
    "latent_upscale": "latent_upscale_models",
    "latent_upscale_models": "latent_upscale_models",
    "refine": "refine",
    "refiner": "refine",
    "interpolate": "frame_interpolation",
    "interpolation": "frame_interpolation",
    "frame_interpolation": "frame_interpolation",
    "video": "video",
    "3d": "3d",
    "model3d": "3d",
}


def _iso_mtime(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _category_for(path: Path, root: Path) -> str:
    try:
        first = path.relative_to(root).parts[0]
        return first
    except Exception:
        return "unknown"


def _folder_tag(path: Path, root: Path, category: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError:
        return "External"
    parts = relative.parts
    if len(parts) >= 3:
        return parts[1]
    if category:
        return category
    return "Root"


def _preview_for_model(path: Path, root: Path) -> str:
    candidates: list[Path] = []
    for suffix in (".png", ".jpg", ".jpeg", ".webp", ".gif"):
        candidates.extend(
            [
                path.with_suffix(path.suffix + f".preview{suffix}"),
                path.with_suffix(f".preview{suffix}"),
                path.with_name(f"{path.stem}.preview{suffix}"),
                path.with_name(f"{path.stem}{suffix}"),
            ]
        )
    preview = next((item for item in candidates if item.exists() and item.is_file()), None)
    if not preview:
        return ""
    return "/model-assets/" + _safe_relative(preview, root)


def _configured_model_roots(settings: NexusSettings) -> list[tuple[Path, str | None]]:
    roots: list[tuple[Path, str | None]] = [(settings.models_dir, None)]
    for alias, paths in settings.model_sources.items():
        category = MODEL_SOURCE_CATEGORY_ALIASES.get(alias.strip().lower())
        for path in paths:
            roots.append((path, category))
    return roots


def scan_models(settings: NexusSettings, include_references: bool = False) -> ModelCatalog:
    roots: list[tuple[Path, str, str | None]] = [
        (root, "managed" if root == settings.models_dir else "reference", category)
        for root, category in _configured_model_roots(settings)
    ]
    if include_references:
        roots.extend((source, "reference", None) for source in settings.reference_model_sources)

    categories: dict[str, list[ModelFile]] = {category: [] for category in MODEL_CATEGORIES}
    supported = {ext.lower() for ext in settings.supported_model_extensions}
    seen_roots: set[tuple[str, str | None]] = set()

    for root, source, category_override in roots:
        if not root.exists():
            continue
        root_key = (str(root.resolve()), category_override)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        for file_path in root.rglob("*"):
            if not file_path.is_file() or file_path.suffix.lower() not in supported:
                continue
            category = category_override or _category_for(file_path, root)
            folder = _folder_tag(file_path, root, category)
            item = ModelFile(
                name=file_path.name,
                path=str(file_path),
                relative_path=_safe_relative(file_path, root),
                category=category,
                folder=folder,
                extension=file_path.suffix.lower(),
                size_bytes=file_path.stat().st_size,
                modified=_iso_mtime(file_path),
                tags=sorted({category, folder, file_path.suffix.lower().lstrip(".")}),
                source=source,  # type: ignore[arg-type]
                preview=_preview_for_model(file_path, root) if source == "managed" else "",
            )
            categories.setdefault(category, []).append(item)

    for items in categories.values():
        items.sort(key=lambda item: (item.folder.lower(), item.name.lower()))

    total = sum(len(items) for items in categories.values())
    return ModelCatalog(root=str(settings.models_dir), categories=categories, total_files=total)


def scan_custom_nodes(settings: NexusSettings, include_references: bool = False) -> list[CustomNodeSummary]:
    roots = [settings.custom_nodes_dir]
    if include_references:
        roots.extend(settings.reference_custom_node_sources)

    nodes: list[CustomNodeSummary] = []
    seen: set[str] = set()
    for root in roots:
        if not root.exists():
            continue
        for child in root.iterdir():
            if not child.is_dir() or child.name == "__pycache__":
                continue
            key = child.name.lower()
            if key in seen:
                continue
            seen.add(key)
            enabled = not child.name.startswith(".") and ".disabled" not in child.parts
            py_files = len(list(child.rglob("*.py")))
            tags = [part for part in child.name.replace("_", "-").split("-") if part]
            nodes.append(
                CustomNodeSummary(
                    name=child.name,
                    path=str(child),
                    enabled=enabled,
                    version=_custom_node_version(child),
                    tags=tags,
                    python_files=py_files,
                )
            )
    nodes.sort(key=lambda item: item.name.lower())
    return nodes


def _custom_node_version(path: Path) -> str:
    git_dir = path / ".git"
    head_path = git_dir / "HEAD"
    try:
        if head_path.exists():
            head = head_path.read_text(encoding="utf-8", errors="ignore").strip()
            if head.startswith("ref:"):
                ref = head.split(":", 1)[1].strip()
                commit_path = git_dir / ref
                commit = commit_path.read_text(encoding="utf-8", errors="ignore").strip() if commit_path.exists() else ""
                branch = Path(ref).name
                return f"{branch}@{commit[:7]}" if commit else branch
            if head:
                return head[:7]
    except Exception:
        pass
    for marker in ("pyproject.toml", "package.json", "requirements.txt"):
        candidate = path / marker
        if candidate.exists():
            try:
                return f"{marker}:{_iso_mtime(candidate)[:10]}"
            except Exception:
                return marker
    return ""


def ensure_model_tree(settings: NexusSettings) -> list[str]:
    created: list[str] = []
    for category in MODEL_CATEGORIES:
        path = settings.models_dir / category
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
    for folder in CHECKPOINT_PRESET_FOLDERS:
        path = settings.models_dir / "checkpoints" / folder
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
        lora_path = settings.models_dir / "loras" / folder
        lora_path.mkdir(parents=True, exist_ok=True)
        created.append(str(lora_path))
    return created
