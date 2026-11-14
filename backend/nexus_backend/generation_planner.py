"""Read-only logical asset planning built on a bounded model catalog."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .asset_resolver import resolve_generation_assets_from_catalog
from .config import NexusSettings
from .schemas import GenerateRequest
from .scanner import scan_models_bounded


@dataclass(frozen=True)
class LogicalAssetPlan:
    """Advisory asset selection; never means the full generation is ready."""

    status: str
    assets: Mapping[str, str]
    generation_ready: bool
    catalog_complete: bool
    truncated: bool
    roots_checked: int
    entries_scanned: int
    files_indexed: int
    note: str = (
        "Logical model selection only. No media was staged, no Comfy process "
        "was started, no model was loaded, and no prompt was submitted."
    )


def plan_generation_assets(
    settings: NexusSettings,
    request: GenerateRequest,
) -> LogicalAssetPlan:
    """Resolve preset-aware logical model names from a bounded, read-only catalog.

    Partial catalogs may produce tentative selections, but they are never
    promoted to a complete result. The supplied request is not mutated.
    """
    scan = scan_models_bounded(settings, include_references=False)
    assets = resolve_generation_assets_from_catalog(
        scan.catalog.categories,
        request.model_copy(deep=True),
    )
    if not scan.complete:
        status = "partial" if assets else "unverified"
    elif assets:
        status = "selected"
    else:
        status = "no_assets"
    return LogicalAssetPlan(
        status=status,
        assets=MappingProxyType(dict(assets)),
        generation_ready=False,
        catalog_complete=scan.complete,
        truncated=scan.truncated,
        roots_checked=scan.roots_checked,
        entries_scanned=scan.entries_scanned,
        files_indexed=scan.files_indexed,
    )
