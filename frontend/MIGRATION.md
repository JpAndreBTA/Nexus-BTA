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
