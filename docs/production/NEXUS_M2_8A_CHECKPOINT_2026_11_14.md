# Nexus-BTA — M2.8A: bounded model-source presence audit

**Date:** 2026-11-14 (America/Sao_Paulo)  
**Project:** JpAndreBTA/Nexus-BTA — separate from VEYLUM and Game3  
**Branch:** `audit/nexus-modernization-m1-2026-11`  
**PR:** #17, draft only; do not merge to `main`.  
**Predecessor:** `docs/production/NEXUS_M2_7_CHECKPOINT_2026_11_14.md`

## Delivered

M2.8A extends the manual M2.7 Studio preflight with a bounded, read-only audit of model filenames referenced by the **patched API workflow graph**.

### Audit implementation

- New `backend/nexus_backend/model_source_audit.py`.
- It extracts file-like values from known Comfy model loader inputs (checkpoint, UNet/diffusion, VAE, text encoder/CLIP, LoRA, ControlNet, upscale). Unknown loader fields with a supported model extension are conservatively searched across known model categories.
- It checks only roots that participate in Nexus's current asset-resolution model: `settings.models_dir/<category>` and matching `settings.model_sources[category]`. It deliberately excludes `reference_model_sources`, because the production `resolve_generation_assets(...)` calls `scan_models(..., include_references=False)`.
- Read-only metadata only: it checks directory entries and file names; it never opens model weights or reads their contents, computes hashes, imports checkpoints, downloads files, launches Comfy, or touches VRAM.
- Symlink files/directories and symlink roots are not traversed. Every inspected path is resolved and checked to remain inside its configured root before it can count as a match.
- Strict budgets: 24 model references, 64 source roots, 20,000 directory entries total, depth 8, at most 4 source labels per reference. Exceeding a bound marks the result partial/unverified rather than claiming absence.
- The report does not echo filenames, absolute paths, sizes or mtimes. It includes node ID/class/input, model category, status, match count, generic source labels and scan counts.
- Status semantics:
  - `found`: one matching physical file in a scanned configured root.
  - `multiple_matches`: multiple physical files share the requested name; the result is ambiguous and requires review.
  - `not_found`: every relevant configured root was successfully scanned and no matching file was found.
  - `unverified`: a root is missing/inaccessible, a symlink/depth/budget restriction prevented complete search, no relevant root is configured, or the supplied reference is absolute/unsafe.
- The manual `POST /api/comfy/workflow-preflight` invokes this audit on the patched in-memory graph. It remains local-only and never submits a generation prompt.
- Studio now surfaces model findings even when Comfy is offline. Missing, ambiguous, unverified or truncated model results change the heading to **Workflow preflight needs review**, even when the node ABI itself is statically valid.

### Explicit scope limitations

This is a **filename-presence audit**, not model integrity or runtime compatibility:
- It does not read or validate safetensors/ckpt/gguf/onnx headers, metadata, tensor keys or hashes.
- It does not load the model or prove GPU/VRAM compatibility.
- It checks model references visible in the patched saved API graph only. It does not yet resolve all automatically selected preset dependencies or reproduce the full dispatch path. Those dependencies may be injected later by `resolve_generation_assets`, preset-specific builders, uploaded references, LoRAs and Director preparation.
- A result can be `found` while Comfy's active object_info menu does not advertise the model, and vice versa; filesystem and live registry evidence are intentionally separate.
- `reference_model_sources` are not searched because the generation asset resolver does not currently use them.

## Tests and gates

New Python tests cover:
- Exact relative-path match in configured model roots and no filename/absolute-path leakage.
- Additional configured category sources.
- Empty vs missing roots, absolute/unsafe references, depth cap and exact 24-reference cap.
- Symlink directory and symlink file handling without following the target.
- Multiple physical matches vs duplicate labels for one physical file.
- Preflight callback receives the patched clone, not the saved workflow; HTTP contract includes bounded model-source summary.
- Chromium mocked acceptance opens the model audit disclosure, sees a found/not-found example and verifies no `/api/generate` submission.

The previous M2.7 complete run passed 4/4 at `25650900`. M2.8A's latest full run for `6a824e08` must be checked before declaring this milestone fully homologated. The immediately preceding UI/model-report run `15eeb929` passed 4/4; the newest commit adds stricter symlink/duplicate tests and error handling.

## Next ideal substage — M2.8B: shared generation dispatch planner

Do not try to make the existing `_run_generation_core` itself a dry-run by calling it with flags: it currently mixes planning with stateful actions (runtime option application, potential restart/free-memory, runtime auto-start, media staging/conversion, optional node ensure/install/restart, job state and eventual queue). A safe planner must be extracted in deliberate layers:

1. Create immutable input snapshot and pure request normalization/validation helpers.
2. Separate pure asset resolution from filesystem staging. Model resolver must expose selected logical asset IDs plus source evidence, while staging/upload remains a later explicit phase.
3. Extract preset workflow selection/building/patching into a shared pure builder used by both the live dispatcher and dry-run, with placeholders for reference/mask/video files that have not been staged yet.
4. Make side-effect boundaries explicit and typed: filesystem writes, Comfy startup/restart, VRAM release, custom-node install, upload, prompt queue. Planner must not call any of them.
5. Add parity tests: for workflows with no staged media and a mocked object_info, the plan graph must equal the live dispatcher graph immediately before queueing. Unsupported complex routes must return `unverified/unsupported`, never a false "ready".
6. Only after parity gates, add custom-node source/version metadata and compare it with live schema fingerprints. Do not infer package version from timestamps alone.

## Other pending work

- Real Windows + installed ComfyUI/GPU QA across SD/SDXL, Flux, Qwen/Ideogram4, WAN/LTX Director, Music3, LoRA/Civitai, inpaint/outpaint/ControlNet and 3D.
- Custom-node version/source manifest and node-to-module mapping; current Comfy schema signature is not a custom-node package version.
- Full textured/multi-mesh GLB preview, long-session 4K/8K mask memory/VRAM testing, Gallery SQLite/cache concurrency, moderate npm advisories and /app feature parity.
- No remote exposure. The bearer-token/read-only route gate is not a substitute for an authenticated HTTPS gateway, roles, trusted proxy policy, token lifecycle and rate limits.
- Keep legacy `/ui`, default localhost, custom workflows and user settings intact. No automatic dependency, Comfy, CUDA, model or custom-node updates.

## Continuity commands

```powershell
git fetch origin
git switch audit/nexus-modernization-m1-2026-11
git pull --ff-only
cd frontend
npm ci
npm run check
npm run test:unit
```

Before M2.8B, inspect `_run_generation_core` and each helper's side effects; do not duplicate the large preset switch into a second implementation.
