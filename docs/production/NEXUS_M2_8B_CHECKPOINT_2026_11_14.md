# Nexus-BTA — M2.8B: shared workflow graph builder and model-source preflight

**Date:** 2026-11-14 (America/Sao_Paulo)  
**Project:** JpAndreBTA/Nexus-BTA — independent from VEYLUM and Game3  
**Branch:** `audit/nexus-modernization-m1-2026-11`  
**PR:** #17 remains draft; do not merge to `main`.  
**Predecessors:** `NEXUS_M2_8A_CHECKPOINT_2026_11_14.md`, `NEXUS_M2_7_CHECKPOINT_2026_11_14.md`.

## Delivered

### M2.8A — bounded model-source presence audit

- New `backend/nexus_backend/model_source_audit.py` checks model filename references in the patched API graph against the same managed/category roots used by `resolve_generation_assets`: `models_dir/<category>` and matching `model_sources[category]`.
- Does not inspect `reference_model_sources` because current generation asset resolution excludes those roots.
- No model file contents, headers, tensors, hashes, weights or metadata are opened. No model is loaded, no VRAM is touched, no download or Comfy process action is performed.
- Symlink roots, files and directories are skipped. Resolved paths must remain inside the configured root. Bounds: 24 references, 64 roots, 20,000 directory entries, depth 8, four source labels per model.
- `found`, `multiple_matches`, `not_found` and `unverified` semantics are conservative. Missing roots, permission errors, symlink/depth/scan budgets and unsafe absolute references do not become false "not found" results.
- Results omit model filenames, absolute paths, file sizes and mtimes. The Studio surfaces the findings even while Comfy is offline; missing, ambiguous, unverified or truncated model checks require review.
- Tests include exact nested paths, additional category roots, multiple physical matches, symlink directory/file escapes, depth cap, absolute references, exact reference budget and no path/name leakage.

### M2.8B-1 — shared in-memory workflow graph builder

- New `prepare_workflow_graph(data, request, object_info, assets)` in `backend/nexus_backend/workflows.py`.
- `WorkflowRegistry.load_api_workflow` (live saved-workflow dispatch) now delegates API/UI format conversion and patching to this shared builder.
- `inspect_patched_preview` (manual dry-run) now uses the same builder on its saved API graph and a private deep copy of the request. The graph is cloned before patching.
- This removes the previous split where live dispatch parsed/converts/patches in `WorkflowRegistry`, while the preflight independently called `patch_workflow`.
- Existing live request mutation semantics are preserved: `patch_workflow` may normalize/mutate the request object passed by live dispatch. Dry-run explicitly passes a deep copy. The builder is shared, but it is not yet a fully immutable `GenerationPlan`.

## Verification

- M2.8A final code head `6a824e0`: GitHub Actions run **4/4 passed** (Python, frontend, Rust, Chromium), run ID `43570475700`.
- M2.8B shared-builder head `9edb2e4`: GitHub Actions run **4/4 passed**, run ID `43570719653`.
- Chromium uses a mocked backend and asserts the model-source details render and preflight never submits `/api/generate`. It does not certify real Comfy, model weights, CUDA or GPU behavior.
- Python tests compare the live registry loader's prepared graph with the shared builder for a deterministic saved API workflow and assert the original graph remains unchanged.

## What this does NOT complete

1. The preflight still is not the exact final dispatch graph. `_run_generation_core` continues to perform runtime option application, possible Comfy restart/free-memory, Comfy auto-start, media conversion/staging, dynamic preset builders, optional node ensure/install/restart, job updates, and eventual prompt queueing.
2. The model-source audit sees model file references present in the patched saved graph. It does not yet run the full `resolve_generation_assets` decision tree or list automatically injected preset dependencies that are not in that graph.
3. No custom-node package version drift check is implemented. The Comfy `schema_signature` fingerprints active node schemas, not the installed custom-node package revision.
4. No real Windows + installed ComfyUI + GPU acceptance was possible in this CI.

## Next ideal substage — M2.8B-2: pure asset resolution and dispatch boundary

Proceed in this order; do not duplicate the preset switch:

1. Inspect and classify every helper called by `_run_generation_core` as pure normalization, filesystem read, filesystem write/staging, Comfy process/VRAM action, node installation, or prompt submission.
2. Extract an immutable request snapshot and pure validation/normalization helpers first. Keep the existing request object semantics until callers are explicitly migrated.
3. Split logical model selection from staging. Reuse the existing preset-aware asset resolver rules, but return logical selected asset identifiers and bounded source evidence separately from upload paths.
4. Represent un-staged reference images, masks, videos, Director frames and audio as typed placeholders. A plan must never create temporary files or upload them.
5. Make runtime startup/restart/free-memory, node install/enable, file staging, Comfy upload and prompt queue explicit effectful operations outside the planner.
6. Use a plan/dispatch parity test for at least one deterministic API workflow before expanding preset coverage. Complex presets with unsupported media or dynamic nodes must return `unverified/unsupported`, not "ready".
7. Only after the shared planner can describe the exact selected preset route, add custom-node source/version metadata and correlate module identity to live Comfy schema signatures. Do not infer version from file timestamps alone.

## Other pending work

- Real Windows/ComfyUI/GPU QA: SD/SDXL, Flux, Qwen/Ideogram4, WAN/LTX Director, Music3, LoRA/Civitai, inpaint/outpaint/ControlNet and 3D.
- Full textured/multi-mesh GLB preview; long-session 4K/8K mask memory and VRAM tests; Gallery SQLite/cache concurrency; moderate npm advisories; /app feature parity.
- No remote exposure. Current local bearer/read-only gate is not a replacement for an authenticated HTTPS gateway, roles, trusted proxy policy, token lifecycle and rate limits.
- Preserve `/ui`, default localhost, user workflows, installed models, local presets and runtime settings. No automatic Comfy, CUDA, model or custom-node updates.

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

Read this checkpoint, then M2.8A, M2.7 and the earlier checkpoints listed in `AGENTS.md` before continuing.
