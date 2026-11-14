from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path

from .config import NexusSettings
from .schemas import CustomNodeSummary, ModelCatalog, ModelFile


MODEL_CATEGORIES = [
    "checkpoints",
    "diffusion_models",
    "unet",
    "loras",
    "vae",
    "audio_vae",
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
    "ltx_25",
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


MAX_BOUNDED_MODEL_ROOTS = 64
MAX_BOUNDED_MODEL_ENTRIES = 20_000
MAX_BOUNDED_MODEL_DEPTH = 8
MAX_BOUNDED_MODEL_FILES = 4096


@dataclass(frozen=True)
class BoundedModelCatalogScan:
    catalog: ModelCatalog
    complete: bool
    truncated: bool
    roots_checked: int
    entries_scanned: int
    files_indexed: int


def scan_models_bounded(
    settings: NexusSettings,
    include_references: bool = False,
) -> BoundedModelCatalogScan:
    """Build a bounded catalog without following symlinks or reading model contents.

    This is intended for advisory planning only. The live generation resolver
    continues to use scan_models() so its existing selection behavior is not
    changed by planner limits.
    """
    roots: list[tuple[Path, str, str | None]] = []
    root_keys: set[tuple[str, str | None]] = set()
    truncated = False

    def add_root(path: Path, source: str, category: str | None) -> None:
        nonlocal truncated
        key = (os.path.normcase(os.path.abspath(os.fspath(path))), category)
        if key in root_keys:
            return
        if len(roots) >= MAX_BOUNDED_MODEL_ROOTS:
            truncated = True
            return
        root_keys.add(key)
        roots.append((path, source, category))

    add_root(settings.models_dir, "managed", None)
    for alias, paths in settings.model_sources.items():
        category = MODEL_SOURCE_CATEGORY_ALIASES.get(str(alias).strip().lower())
        for path in paths:
            add_root(Path(path), "reference", category)
            if truncated:
                break
        if truncated:
            break
    if include_references and not truncated:
        for source_root in settings.reference_model_sources:
            add_root(source_root, "reference", None)
            if truncated:
                break

    categories: dict[str, list[ModelFile]] = {category: [] for category in MODEL_CATEGORIES}
    supported = {str(ext).lower() for ext in settings.supported_model_extensions}
    complete = not truncated
    roots_checked = 0
    entries_scanned = 0
    files_indexed = 0
    seen_roots: set[tuple[str, str | None]] = set()
    stop = False

    for root, source, category_override in roots:
        try:
            if root.is_symlink() or not root.exists():
                complete = False
                continue
            root_real = root.resolve(strict=True)
            if not root_real.is_dir():
                complete = False
                continue
        except (OSError, RuntimeError):
            complete = False
            continue

        root_key = (os.path.normcase(os.fspath(root_real)), category_override)
        if root_key in seen_roots:
            continue
        seen_roots.add(root_key)
        roots_checked += 1
        stack: list[tuple[Path, int]] = [(root_real, 0)]

        while stack and not stop:
            current, depth = stack.pop()
            try:
                if current.is_symlink():
                    complete = False
                    continue
                current_real = current.resolve(strict=True)
                if not current_real.is_relative_to(root_real):
                    complete = False
                    continue
                with os.scandir(current_real) as entries:
                    for entry in entries:
                        if entries_scanned >= MAX_BOUNDED_MODEL_ENTRIES:
                            complete = False
                            truncated = True
                            stop = True
                            break
                        entries_scanned += 1
                        try:
                            if entry.is_symlink():
                                complete = False
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                if depth >= MAX_BOUNDED_MODEL_DEPTH:
                                    complete = False
                                    continue
                                stack.append((Path(entry.path), depth + 1))
                                continue
                            if not entry.is_file(follow_symlinks=False):
                                continue
                            extension = Path(entry.name).suffix.lower()
                            if extension not in supported:
                                continue
                            if files_indexed >= MAX_BOUNDED_MODEL_FILES:
                                complete = False
                                truncated = True
                                stop = True
                                break
                            stat = entry.stat(follow_symlinks=False)
                            file_real = Path(entry.path).resolve(strict=True)
                            if not file_real.is_relative_to(root_real):
                                complete = False
                                continue
                            category = category_override or _category_for(file_real, root_real)
                            folder = _folder_tag(file_real, root_real, category)
                            relative_path = _safe_relative(file_real, root_real)
                            item = ModelFile(
                                name=file_real.name,
                                path=str(file_real),
                                relative_path=relative_path,
                                category=category,
                                folder=folder,
                                extension=extension,
                                size_bytes=stat.st_size,
                                modified=datetime.fromtimestamp(
                                    stat.st_mtime, tz=timezone.utc,
                                ).isoformat(),
                                tags=sorted({category, folder, extension.lstrip(".")}),
                                source=source,  # type: ignore[arg-type]
                                preview="",
                            )
                            categories.setdefault(category, []).append(item)
                            files_indexed += 1
                        except (OSError, RuntimeError):
                            complete = False
            except (OSError, RuntimeError):
                complete = False
        if stop:
            break

    for items in categories.values():
        items.sort(key=lambda item: (item.folder.lower(), item.name.lower()))
    catalog = ModelCatalog(
        root=str(settings.models_dir),
        categories=categories,
        total_files=files_indexed,
    )
    return BoundedModelCatalogScan(
        catalog=catalog,
        complete=complete and not truncated,
        truncated=truncated,
        roots_checked=roots_checked,
        entries_scanned=entries_scanned,
        files_indexed=files_indexed,
    )


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
    # Keep a visible per-template root for organization while storing weights
    # under standard Comfy categories for loader compatibility.
    krea2_root = settings.models_dir / "krea2"
    krea2_root.mkdir(parents=True, exist_ok=True)
    created.append(str(krea2_root))
    music3_root = settings.models_dir / "music3"
    music3_root.mkdir(parents=True, exist_ok=True)
    created.append(str(music3_root))
    for category in ("diffusion_models", "text_encoders", "vae"):
        path = settings.models_dir / category / "music3"
        path.mkdir(parents=True, exist_ok=True)
        created.append(str(path))
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
