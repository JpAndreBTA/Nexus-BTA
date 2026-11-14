# Nexus-BTA — M2.7: deterministic patched-template preflight and Comfy node ABI signature

**Checkpoint date:** 2026-11-14 (America/Sao_Paulo).
**Project:** JpAndreBTA/Nexus-BTA (not VEYLUM or Game3).
**Branch:** `audit/nexus-modernization-m1-2026-11`.
**PR:** #17 — draft, no merge to `main`, /ui legacy retained.

Previous: `NEXUS_M2_6_CHECKPOINT_2026_11_14.md` and `AGENTS.md`.

## What was implemented

### In-memory patched-template preflight (explicit action, no generation)

- `backend/nexus_backend/workflow_preflight.py` provides `WorkflowPreflightRequest` (bounded Pydantic scalar fields, forbids extras) and `inspect_patched_preview`. Input is a saved **API-format** workflow JSON, read via WorkflowRegistry with the prior 4 MiB source file cap. No UI-to-API conversion is attempted in this path.
- Calls the **existing** `patch_workflow` on `deepcopy` of the stored node graph and a deep-copied `GenerateRequest`; the random seed -1 is replaced with deterministic 0. The in-memory preview uses explicit preset, model name, prompt/negative prompt, dimensions, sampler, scheduler, steps, CFG, seed and denoise. It does **not** change the saved workflow, model files, database, Windows registry, or Comfy installation.
- Pre- and post-patch structural bounds: at most 512 nodes, at most 128 inputs/node; 64–8192 px dimensions, up to 200 steps, 4096 chars/prompt. Invalid or unknown JSON fields are rejected. The resulting workflow is passed through the M2.6 ABI validator; raw patched graph and prompts are never returned.
- `POST /api/comfy/workflow-preflight` is **local-only** (outside the remote read-only allowlist). It reads the existing Comfy `/object_info` only if running, never calls `ensure_running` or submits `/prompt`. Offline or unavailable runtime is explicitly unverified. Failure responses do not echo model paths or prompts.
- The Studio Node Workflow view has a manual `Preflight selected settings` button; it sends selected scalar fields only, **not images, inpaint masks, LoRA payloads, Director frames or model file bytes**. Result shows changed node/field counts, bounded ABI findings, advisory note and no-success guarantee. Editing the selected settings resets stale preflight results.
- **Scope warning:** The real `main.py` generation path can inject uploaded assets, dynamically build specialized preset workflows, install/enable runtime dependencies after explicit approval, and route through custom nodes. This preflight is *not* the exact eventual prompt, does not validate a GPU dispatch, and is not proof that the model exists. It is an early, safe compatibility signal.

### Comfy node registration and model-choice evidence

- M2.6 `workflow_input_audit.py` now summarizes the current registry's advertised model selections (e.g. checkpoint/UNet/VAE/LoRA name inputs) and counts names **not found in the active object_info choice menu**. This is not a physical filesystem scan and cannot prove a model is missing from every configured model directory. Values/paths are not echoed in the report.
- A 20-hex-character SHA-256 prefix derived from only the current workflow's relevant node input/output schemas is returned as `schema_signature`. Changes in node choices or input/output ABI change the signature, allowing a user/agent to detect a difference between audits. It is not a signed software provenance record or definitive package version.
- React Studio displays unadvertised model selection counts and ABI signature alongside existing missing input/link counts, without leaking raw registry entries.

## Test and security gates

- Python: test cloned graph is unmodified, deterministic scalar patch changes counts, no model name is echoed, oversized/unknown/invalid payload rejected, offline is unverified, malformed workflow ID rejected. Existing API tests inspect the local route with mocked `object_info`; no Comfy startup or prompt submission.
- Chromium: manual button sends only scalar values to `/api/comfy/workflow-preflight`, displays the advisory patch report, and explicitly verifies zero generation-submit requests. Screenshot added as `frontend/artifacts/studio-workflow-preflight.png`.
- Existing frontend unit/typecheck/build + high npm audit, Rust and Python security tests still run. A missing `packaging` wheel in lightweight Python CI was corrected by adding it to the isolated CI install, **not** by auto-installing/altering the user's embedded runtime.
- Verified full source gates: `3e84670` (Python deps), `29e5f6f` (React manual preflight), `559acd44` (model-choice schema signature). Check the latest `319f29f` browser gate on the PR for its result; never assume success merely because the prior head passed.

## Gaps, exclusions, and next M2.8

1. **Real Windows + installed ComfyUI + actual GPU runs** remain mandatory, covering model paths/checkpoints, SDXL, Flux, Qwen/Ideogram4, WAN/LTX Director, Music3, Civitai/LoRAs, inpaint and 3D. CI mocks and static node menus cannot certify it.
2. More precise dispatch dry-run needs a **shared planning function** that produces the *exact* prepared graph without staging files, automatically starting Comfy, restarting nodes, downloading weights or scheduling a prompt. Do not copy substantial generation orchestration blindly into a new endpoint.
3. Add bounded read-only model-source presence checks against explicitly configured roots without following symlink escapes, plus metadata for custom-node ABI version drift. Comfy's model choices alone are insufficient.
4. Full PBR/multi-mesh GLB preview, native 4K/8K mask long-session memory/VRAM QA and full /app vs /ui feature parity are pending.
5. Stress-test caches/SQLite under concurrent gallery reads, investigate moderate npm advisories, and progressively split monolithic backend and Studio modules with API compatibility tests.
6. **Remote deployment is not approved.** The ASGI read-only bearer-token defense is insufficient without an independent authenticated HTTPS gateway, roles, rotation, proxy trust and rate limits. No tunnel or remote listener was activated.
7. PR #17 remains draft, no auto-merge. Do not update Comfy custom nodes, PyTorch/CUDA, model files, Windows registry or user workflow JSON without explicit safety review.

## Continuity

```powershell
git fetch origin
git switch audit/nexus-modernization-m1-2026-11
git pull --ff-only
cd frontend
npm ci
npm run check
npm run test:unit
```

Before advancing to M2.8, check latest run for the checkpoint commit and compare the operator's actual Windows/ComfyUI environment against the tracked schema signatures. Commit only coherent milestones with regression gates.
