# Nexus-BTA — M2.5: local-first remote security gate + workflow registration QA

**Date:** 2026-11-14 (America/Sao_Paulo).
**Branch:** `audit/nexus-modernization-m1-2026-11`.
**PR:** #17, remains draft; `main` and legacy `/ui` unchanged.
**Prior checkpoint:** `docs/production/NEXUS_M2_4_CHECKPOINT_2026_11_14.md`.

## M2.5-A — FastAPI startup registration regression

- Fixed a production startup-order defect: `main.py` previously tried to include the new Comfy capability router before `app = FastAPI(...)`.
- The router is now included after FastAPI creation; `comfy` and workflow registry dependencies remain available before the router is called.
- `backend/tests/test_main_registration.py` uses AST to assert no top-level app route/middleware/mount access occurs before app creation. No side-effect-heavy Comfy/GPU import is required for this regression gate.
- This test checks static registration order, not full Windows runtime startup; manual Windows acceptance is still required.

## M2.5-B — Harden non-loopback listeners without turning on remote use

- `backend/nexus_backend/security.py` now requires **both** `NEXUS_UNSAFE_NETWORK_ACK=I_UNDERSTAND_THE_RISK` and a strong operator-provided `NEXUS_REMOTE_API_TOKEN` before `run_backend.py` can bind to an external interface.
- `backend/nexus_backend/remote_gate.py` adds an ASGI-wide, fail-closed bearer-token check for all non-local HTTP requests, including static mounts.
- It checks configured listener host **and real ASGI socket/peer addresses**. This prevents a direct `uvicorn ... --host 0.0.0.0` launch from bypassing the guard merely because `NEXUS_BACKEND_HOST` is unset. IPv4, IPv6 and IPv4-mapped IPv6 are covered.
- Direct Uvicorn launch also requires the explicit acknowledgment; a token alone is not enough. Only one Authorization bearer header is accepted, compared in constant time. No default token is generated or published.
- After authentication, only **read-only** `GET/HEAD` routes are authorized remotely: `/api/gallery/page`, `/api/gallery/thumb/{relative_path}`, `/api/gallery/video-thumb/{relative_path}` and `/api/comfy/capabilities`. Mutations, workflow installation, configuration endpoints, direct output/input/model files, frontend pages and WebSocket connections remain forbidden.
- Default local `127.0.0.1`/`localhost` desktop mode is unchanged and needs no token. Browser-based remote UI is **not** enabled by these changes.
- Security tests cover missing/wrong bearer tokens, ack-only and token-only failures, direct Uvicorn peer detection, denied static files, denied installs/process start, read-only allowlist and local compatibility.
- **Threat model limitation:** This is an experimental defense in depth, NOT a complete authenticated HTTPS reverse proxy and NOT authorization for public access. Raw HTTP could expose a bearer token; never open the service to the internet/LAN on the strength of this middleware alone. Any reverse proxy that forwards traffic to loopback requires its own independent authentication, TLS, rate limits and access controls. There are no remote login sessions, user roles or token-rotation UI.

## M2.5-C — Check exact workflow node classes without running ComfyUI

- `GET /api/comfy/workflow-capabilities?workflow_id=<id>` obtains the registered class list from the actual `WorkflowRegistry.find + summarize` (API/UI workflow file), not a synthetic graph in React.
- Reads `/object_info` only when Comfy is already running; no auto-start, dependency installation or node upgrades. A failed/missing/offline registry is reported as **unverified**, never as proof that a node is absent.
- Bounded by a strict workflow identifier and a maximum of 256 unique classes. Empty, oversized or uninspectable class sets are not falsely marked compatible. Missing class names are returned to the local Studio for diagnostics.
- React Studio Node Workflow view now fetches capability results by workflow ID, supports partial/unverified statuses and reminds users that registered classes alone do not prove valid inputs, model assets, custom node ABI, VRAM or successful execution.
- Python FastAPI HTTP tests cover found/missing/offline, invalid workflow ID, large workflows and no provider call on invalid input. A new Chromium smoke fixture creates a read-only mock workflow and verifies offline UI behavior and exact ID-based request.
- This is **registration compatibility** only, not the full SD/Flux/WAN/LTX/Music3 runtime execution matrix.

## CI gates and evidence

- All commits remain on `audit/nexus-modernization-m1-2026-11`, never on main.
- Code commits include FastAPI order fix `7b3e911`, non-loopback gate `7ccfe07`, workflow checks `ba1e4af`/`6331e71`, direct-server hardening `fcabaa4`, Chromium mock `2bcfe4b`.
- Four jobs: Python security/API/registration tests, React TypeScript + unit + build/npm high audit, Rust tests, and real Chromium with mocked backend.
- At the time this checkpoint was drafted, the previous code gate at `fcabaa4` was all green (`43553481562`), with the latest Chromium acceptance run for `2bcfe4b` still to be inspected. Never substitute prior success for latest HEAD success.
- The browser smoke does **not** launch ComfyUI, GPU, install models or test local Windows assets.

## Known issues / next M2.6

1. Windows/ComfyUI/GPU real acceptance remains pending for SD/SDXL, img2img/inpaint/ControlNet, Flux/Qwen/Ideogram, WAN/LTX/Director, Music3, LoRAs and 3D. Do not merge without this QA.
2. A production remote gateway still needs TLS termination, trusted-proxy policy, real authentication/authorization, session/credential lifecycle, rate limiting and hardened streaming before any network exposure.
3. GLB viewer remains a first-mesh geometry-only prototype. Full PBR, multi-mesh hierarchy, animations and textures require separate audited implementation.
4. Inpaint requires native source-resolution mapping QA, extreme-aspect-ratio fixes, and long-session RAM/GPU profiling; the backend nearest-scales masks to requested resolution for some presets, which is not global proof of 1:1 alignment.
5. Class registration is only one factor in Comfy compatibility. Next develop bounded asset/model/dependency input checks with no automatic installations, plus versioned node ABI compatibility gates.
6. Four moderate npm audit advisories remain; avoid unreviewed blanket upgrades.
7. Decompose monolithic `main.py` and `workflows.py` incrementally, keep wire contracts, do not replace `/ui` until parity.
8. Gallery SQLite scan/pagination and WebP/video caches need real disk/concurrency/long-session stress testing.

## Local continuity

```powershell
git fetch origin
git switch audit/nexus-modernization-m1-2026-11
git pull --ff-only
cd frontend
npm ci
npm run check
npm run test:unit
```

No Windows registry edits, ComfyUI/driver/node upgrades, model downloads or persistent background tasks were performed in this milestone. Keep commit, CI and checkpoint documentation updated before any further production.