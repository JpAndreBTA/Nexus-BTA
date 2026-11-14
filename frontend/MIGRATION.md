# Nexus BTA Frontend Migration

This folder is the new responsive web app track. The legacy `/ui` remains served from `index.html` until React reaches feature parity.

## Stack

- React + TypeScript
- Rsbuild
- TanStack Router
- TanStack Query
- Zustand

## Token Budget Rules

- Prefer typed API contracts in `src/api/types.ts` over rereading backend payloads.
- Prefer small stores in `src/stores/*` over large global state files.
- Keep migration notes here and in concise Serena memories.
- Migrate one workflow at a time; do not rewrite `index.html` during React work unless compatibility requires it.

## Route Plan

- `/` studio shell and backend status
- `/extras` first full migrated tool
- `/models` catalog/manager foundation
- `/ui` legacy bridge

## Backend Integration Plan

- During development, run the React app on `http://127.0.0.1:3000` and proxy `/api`, `/outputs`, `/assets` and `/model-assets` to the FastAPI backend.
- When a stable build exists, serve it from a new backend prefix such as `/app`.
- Keep `/ui` and `/index.html` on the legacy `index.html` until React reaches feature parity.
- Do not mount the React SPA at `/`; that could interfere with `/api`, `/outputs` and existing root status endpoints.
- For the production backend build, configure the frontend asset base for `/app/` to avoid colliding with the legacy `/assets` mount.

## First Migration Slice

Extras moves first because the backend already exposes file upload jobs, polling and output metadata through `/api/extras/*`.

The legacy UI can also be migrated through small React islands if a full page rewrite becomes too risky. Candidate islands: LoRA modal, model catalog, Extras controls, then Gallery.

## Checkpoint: Extras Functional Shell

Implemented in React:

- Source upload and drag/drop for image, video and image sequence candidates.
- Plan builder for image upscale, video interpolate/upscale/encode and model-based remove background.
- Model selectors populated from `/api/models`.
- Multipart start through `/api/extras/start`.
- Polling through `/api/extras/{job_id}`.
- Image/video preview using backend `/outputs` URLs.
- Responsive controls and action bar.

Smoke tested:

- Image alpha upscale: `RGBA 160x120 -> RGBA 320x240`.
- Remove BG image with BiRefNet: `RGBA 1920x1080`, alpha extrema `(0, 255)`.
- Video interpolate + upscale: `1920x1080 24fps -> 3840x2160 60fps`, 95 frames.

Screenshots:

- `test-results/react-extras-image-smoke.png`
- `test-results/react-extras-removebg-smoke.png`
- `test-results/react-extras-video-smoke.png`

## Checkpoint: Backend `/app` and Gallery

Implemented:

- FastAPI serves the React production build under `/app`.
- React static assets are mounted under `/app/static`.
- Legacy `/ui` and `/index.html` remain unchanged.
- TanStack Router detects `/app` as runtime basepath.
- Gallery route reads `/api/gallery`, renders a responsive grid and preview panel.
- Gallery item preview can send an existing `/outputs/...` URL to Extras through `source_url`, avoiding reupload.

Smoke tested:

- `/app` renders Studio through backend port `7861`.
- `/app/extras` renders Extras through backend port `7861`.
- `/app/gallery` loaded 112 output items.
- Gallery -> Extras handoff loaded an output source.
- Extras processed a remote `/outputs/...` source via backend `source_url`.

Screenshots:

- `test-results/backend-app-home.png`
- `test-results/backend-app-extras.png`
- `test-results/backend-app-gallery.png`
- `test-results/backend-app-gallery-to-extras.png`
- `test-results/backend-app-extras-remote-source-smoke.png`

## Checkpoint: Studio Basic Generation Shell

Implemented:

- `/app` Studio now has a React txt2img form instead of only a placeholder.
- Typed generation request/job contracts in `src/api/types.ts`.
- Generation API methods for `/api/generate/start`, `/api/generate/{job_id}` and cancel.
- Zustand generation form state in `src/stores/generationStore.ts`.
- Prompt, negative prompt, preset, model, size, steps, cfg, sampler, scheduler and seed controls.
- Job polling, cancel action, progress overlay and output preview.

Validation:

- `npm run check` passed with `NEXUS_FRONTEND_BASE=/app/`.
- `/app` renders Studio through backend in desktop and mobile.
- Default preset is `SD` to avoid Anima-specific smoke requirements.

Screenshots:

- `test-results/backend-app-studio-desktop.png`
- `test-results/backend-app-studio-mobile.png`

## Checkpoint: Studio Model Filtering and LoRA Base

Implemented:

- Studio model select filters models by preset with fallback to all models.
- `/api/loras` client/query/types.
- Zustand LoRA store with active LoRAs, strength update and removal.
- Studio LoRA search, add/remove and active strength controls.
- Generation payload now sends active LoRAs using backend-compatible fields.

Validation:

- `NEXUS_FRONTEND_BASE=/app/ npm run check` passed.
- `/app` loaded 24 LoRA candidates in the initial list.
- Adding one LoRA creates one active LoRA row.

Screenshot:

- `test-results/backend-app-studio-lora.png`

## Checkpoint: Settings and Remote Readiness Shell

Implemented:

- `/settings` route in React.
- Runtime cards for backend, Comfy status, model/custom node paths.
- Access-mode checklist for local, LAN and internet usage.
- Security gate notes before any public internet exposure.

Validation:

- `NEXUS_FRONTEND_BASE=/app/ npm run check` passed.
- `/app/settings` renders through backend in desktop and mobile.

Screenshots:

- `test-results/backend-app-settings-desktop.png`
- `test-results/backend-app-settings-mobile.png`

## Checkpoint: Repeatable App Smoke

Implemented:

- `frontend/scripts/smoke_app.py` checks `/app`, `/app/extras`, `/app/gallery`, `/app/models` and `/app/settings`.
- Each route is validated on desktop and mobile viewports and writes screenshots to `test-results/app-smoke-*.png`.

Validation:

- Python compile passed for backend and smoke script.
- `NEXUS_FRONTEND_BASE=/app/ npm run check` passed.
- `python frontend/scripts/smoke_app.py` passed all 10 route/viewport checks.

## Checkpoint: Models Catalog Upgrade

Implemented:

- `/models` now has category filtering, search, detailed model panel, size formatting and tags.
- Layout is responsive with category/detail panels collapsing on tablet/mobile.

Validation:

- `NEXUS_FRONTEND_BASE=/app/ npm run check` passed.
- `frontend/scripts/smoke_app.py` passed all desktop/mobile route checks.

## Checkpoint: Studio Simple Img2Img

Implemented:

- Studio mode switch for `txt2img` and `img2img`.
- Reference image upload and local preview.
- Denoise slider for img2img.
- Generation payload now includes `img2img`, disabled `controlnet`, `director` and `runtime` fields.

Validation:

- `NEXUS_FRONTEND_BASE=/app/ npm run check` passed.
- Img2img smoke loaded `sample.png` as a data URL reference.
- `frontend/scripts/smoke_app.py` passed all desktop/mobile route checks.

Screenshot:

- `test-results/backend-app-studio-img2img.png`

## Checkpoint: Studio Inpaint, ControlNet and Workflow Graph

Implemented:

- Studio img2img mode now supports `Image` and `Inpaint`.
- Inpaint canvas exports a PNG data URL mask using the backend-compatible red/white mask route.
- Inpaint controls include brush size, erase, clear, denoise, resize mode, mask blur and fill mode.
- ControlNet panel sends the backend-compatible `controlnet` payload, with catalog-filtered models for SD/SDXL/LTX.
- Generation state now persists lightweight settings and active workflow selection without persisting image/base64 sources.
- New `/workflow` React route lists backend workflows, loads workflow analysis, renders nodes/wires, reports missing nodes/dependencies and can activate a workflow for Studio.
- `frontend/scripts/smoke_app.py` now includes `/app/workflow` in desktop/mobile coverage.
- `frontend/scripts/smoke_generation_contract.py` intercepts `/api/generate/start` and verifies inpaint mask, ControlNet image and active workflow payload fields without running a heavy model.

Validation:

- `NEXUS_FRONTEND_BASE=/app/ npm run check` passed.
- Inpaint/ControlNet Playwright smoke drew real mask pixels and loaded a ControlNet image thumbnail.
- Workflow smoke loaded graph nodes, activated a workflow, reloaded Studio and confirmed the active workflow survived.
- Generation contract smoke passed for inpaint + ControlNet + workflow_id.
- `frontend/scripts/smoke_app.py` passed all 12 desktop/mobile route checks.

Screenshots:

- `test-results/backend-app-studio-inpaint-controlnet.png`
- `test-results/backend-app-studio-controlnet.png`
- `test-results/backend-app-workflow-graph.png`
- `test-results/app-smoke-generation-contract.png`

## Checkpoint: M1 modernization candidate (2026-11-12)

- Six React routes are lazy loaded instead of downloading the Studio as part of the initial shell.
- Appearance settings now persist under the existing `nexus-ui-state` key (theme/accent/density/motion).
- Civitai picker no longer lists imaginary search results and includes actual filters and download progress.
- CORS is restricted; accidental LAN/tunnel use is blocked by default until an explicit unsafe-network acknowledgment. This does **not** provide remote authentication.
- Rust mask core is an isolated experimental crate, not enabled in production.
- Reference: `docs/audit/NEXUS_UI_AUDIT_2026_11_12.md`.
- Gate remains: CI checks, browser smoke with actual backend, Windows + ComfyUI work, visual confirmation. Do not redirect /ui to /app yet.

## Checkpoint: M2 UI modules, bounded inpaint, layout presets and browser-smoke

See `docs/production/NEXUS_M2_CHECKPOINT_2026_11_12.md` for the completed commits, accepted tests, remaining regressions and next milestone. Do not mark legacy `/ui` deprecated yet.

## M2.2 checkpoint — 13 November 2026

See `docs/production/NEXUS_M2_2_CHECKPOINT_2026_11_13.md`. New paginated React Gallery uses `/api/gallery/page` with an SQLite index; `/api/gallery` and `/ui` are intentionally retained. The Studio generation request serializer and validation are modular and unit-tested. Search and sort remain local to loaded Gallery pages. Do not mark /app feature complete.

## M2.3 checkpoint — 13 November 2026

See `docs/production/NEXUS_M2_3_CHECKPOINT_2026_11_13.md`. This supersedes M2.2's limitation that search/sort applied only to loaded pages: the new Gallery searches filenames/folders and sorts/filters over the full SQLite index. Image tiles use safe bounded on-demand WebP thumbs. Source/mask share a zoomable/pannable inpaint viewport. Video/3D previews, full-resolution mask export mapping and real Windows/ComfyUI tests remain pending.

## M2.4 checkpoint — 14 November 2026

See `docs/production/NEXUS_M2_4_CHECKPOINT_2026_11_14.md`. The Gallery now supports on-demand bounded local FFmpeg WebP video posters and an experimental lazy GLB first-mesh geometry view. The Studio Node Workflow mode performs read-only ComfyUI registration checks without auto-starting or installing nodes. This does not certify full PBR 3D assets, model generation or Windows GPU parity. Legacy `/ui` and default routes remain intact.

## M2.5 checkpoint — 14 November 2026

See `docs/production/NEXUS_M2_5_CHECKPOINT_2026_11_14.md`. FastAPI capability router registration order corrected with structural test. Non-loopback requests now require an explicitly configured strong token AND unsafe-network acknowledgment; all remotely exposed paths are read-only allowlisted, other routes and mounts denied. This is not permission to publish Nexus on a network or a replacement for an authenticated HTTPS gateway. Studio workflow capability checks now derive exact class names from stored workflow JSON, not synthetic graph nodes; offline remains unverified. Legacy `/ui` and default local mode are preserved.

## M2.6 checkpoint — 14 November 2026

See `docs/production/NEXUS_M2_6_CHECKPOINT_2026_11_14.md`: a bounded read-only static Comfy API workflow input/link/choice report is now available in Studio, with explicit unverified states and no automatic installs. Inpaint emits a native-source-resolution opaque binary PNG mask up to a 16 MP / 8192px memory cap and otherwise falls back to editor resolution with an explicit warning. Real Windows/Comfy/GPU behavior is not certified by Chromium mock acceptance; keep /ui, default localhost and PR draft.

## M2.7 checkpoint — 14 November 2026

See `docs/production/NEXUS_M2_7_CHECKPOINT_2026_11_14.md`. The Node Workflow view includes an explicit manual scalar-only patched-template preflight; its preview is performed on copies without Comfy prompt submission or file uploads and does **not** reproduce full dispatch staging. Node schema ABI signature and model menu visibility are exposed as bounded privacy-preserving diagnostics, not claims of physical weight availability. `/ui` and localhost default remain unchanged; PR #17 stays draft.

## M2.8A checkpoint — 14 November 2026

See `docs/production/NEXUS_M2_8A_CHECKPOINT_2026_11_14.md`. The manual patched-template preflight now performs bounded, read-only filename checks against the same configured managed/category roots used by Nexus asset resolution. It never opens model weights or follows symlinks; incomplete scans remain unverified. The audit covers model file references in the patched graph only, not auto-resolved preset dependencies or the exact dispatch graph. M2.8B is the shared pure generation-planning extraction; keep `/ui`, localhost defaults and PR #17 draft.

## M2.8B checkpoint — 14 November 2026

See `docs/production/NEXUS_M2_8B_CHECKPOINT_2026_11_14.md`. Model-source presence auditing is bounded, read-only and symlink-safe; the live saved-workflow loader and manual preflight now share `prepare_workflow_graph` for API/UI conversion and patching. This is the first shared-planner extraction, not yet a complete dispatch plan: model auto-resolution, media staging, runtime effects, node installation and prompt submission remain outside it. Keep `/ui`, localhost defaults and PR #17 draft.

M2.8B-2 further separated pure preset-aware logical asset selection into `resolve_generation_assets_from_catalog`; the live resolver remains a filesystem catalog adapter and preserves existing selection rules. The planner must use a future bounded catalog producer rather than calling the current unbounded `scan_models` scan.
