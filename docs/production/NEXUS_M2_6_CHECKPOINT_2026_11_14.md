# Nexus-BTA — M2.6: static Comfy input ABI + native-resolution inpaint masks

**Date:** 2026-11-14. **Repository:** JpAndreBTA/Nexus-BTA.
**Working branch:** audit/nexus-modernization-m1-2026-11.
**PR:** #17 (draft). **Main and legacy /ui remain untouched.**
Continue after M2.5 and AGENTS.md; do not reuse VEYLUM or Game3 assets/code.

## Delivered — bounded, read-only ComfyUI input compatibility

- Added `backend/nexus_backend/workflow_input_audit.py` as a pure static diagnostics module. The stored workflow JSON is read with a 4 MiB maximum; API graph inspection accepts at most 512 nodes and at most 128 inputs per node. A maximum of 32 findings are returned, with explicit truncation.
- For saved **API-format** workflows, compare class registrations and required/optional input schema definitions in the current `/object_info` registry. Validate missing node classes, broken node links, missing/invalid output slots, confirmed output type disagreement, potentially missing required inputs, and literal option values not in the registry choice lists.
- **Important:** Nexus can inject/patch input values before dispatching a workflow. Missing required fields and choice mismatches from the saved unpatched graph are reported as **review items**, not definitive failures. `static_valid` means only that the inspected saved JSON has no recognized static issues; `ready` stays `false` because GPU/models/Comfy execution have not been verified.
- UI-format workflows are explicitly classified as `ui_format`: widgets and subgraphs need runtime conversion before their input ABI can be checked. Offline/connection failures remain unverified, never falsely classed as absent nodes.
- Added `GET /api/comfy/workflow-inputs?workflow_id=...` through the existing `comfy_capabilities.py` router, using the actual `WorkflowRegistry.find` path. It reads files and metadata only; no generation, installs, node manager, Comfy restart or model downloads. It is **not in the non-loopback remote allowlist** and is therefore local-only.
- The Studio Node Workflow view now includes a separate `StudioWorkflowInputStatus.tsx` panel alongside the previous node registration report, showing counts and a bounded list of findings. API client and React Query requests are typed and keyed by workflow ID, with cancellation support.
- Python regression tests cover valid API graph, invalid/missing classes, broken links, review-only cases, offline/UI/oversized inputs and on-disk bounds. FastAPI tests verify only valid workflows reach `/object_info`; Chromium uses a mocked workflow and confirms offline is not mislabeled as compatible.

## Delivered — source-pixel mapping for inpaint mask export

- Existing inpaint UI still edits masks on a bounded canvas of at most 1536 px on the longest side, with approximately 32 MiB undo/redo snapshot budget, independent of source size.
- Added `planMaskExport` in `maskPixels.ts`: when the source image fits **16,777,216 pixels and 8192px on either edge**, the opaque black/white PNG mask is exported at the **source image's original dimensions**. When the source exceeds that bound, the existing editor-sized mask is exported instead and the UI reports the cap.
- The export uses nearest-neighbor resampling of the editor's binary black-and-white mask. This avoids introducing interpolated gray edges while mapping the painted grid to source pixels. The temporary full-size canvas is released after serialization, and full-resolution undo snapshots are not retained.
- The UI reports original, editor, and output mask resolutions and prohibits painting before the reference dimensions have loaded.
- Unit tests cover the mapping for 4096×2160, 7680×4320 (bounded fallback) and portrait 3000×5000; Chromium with a generated 2048×1024 reference checks that the actual exported PNG is exactly 2048×1024 and includes painted white / untouched opaque black areas. Browser test screenshot artifact: `inpaint-native-mask.png`.
- This does **not** yet certify Comfy's actual generation scaling for every resize mode, source aspect ratio, mask orientation, Photoshop-like brushes or 8K memory safety. Input reference and final generation dimensions can differ, and some workflows scale masks further internally. Full Windows/GPU QA is mandatory.

## Gates and evidence

- Earlier commits in this milestone: `7e5b68d` (pure ABI audit), `1aa2cec` (read-only API), `6b827ea` (HTTP tests), `8f3a56e` (Studio panel), `a9184a6` (source-size mask export), `c159c64` (browser native resolution), `941d7a7` (pixel-value browser acceptance).
- First six M2.6 code commits passed GitHub CI independently, including Python, frontend unit/typecheck/build + npm high/critical audit, Rust, and Chromium with mocked API. Latest verification run for the final pixel-value browser assertion needs to be checked on the PR before claiming full milestone approval.
- No ComfyUI/GPU execution or Windows system changes were performed through the cloud CI. No model, workflow JSON, user settings or registry modifications occurred.
- The PR remains in draft; `main` is not merged.

## M2.7 suggested next priorities

1. Windows + ComfyUI/GPU evidence: test the real source-to-mask reference alignments, SDXL, Flux/Qwen/Ideogram 4, WAN/LTX Director, Music3, Civitai/LoRAs and 3D generation. Compare visual output, job errors, VRAM/RAM and per-node `object_info` against actual installed local assets.
2. Static capability audit v2: validate generated/patched API payloads under a safe dry-run mode (no Comfy prompt submission), including built-in node defaults, asset path availability and unsupported custom node versions.
3. Stress-test native mask and thumbnail cache memory usage, 4K/8K/portrait, invalid source images and zoom/pan during long editing sessions. Move to binary mask file upload only with verified backend schema/migration.
4. Full PBR/multi-mesh GLB viewer with bounded GLTF parsing or separately lazy local renderer, without leaking assets or loading external URIs.
5. Continue breaking up `main.py`, `workflows.py` and `HomePage.tsx`, preserving the wire-format contracts, security gates, and legacy /ui feature parity.
6. Before any remote service: genuine trusted TLS authenticated gateway, roles, token rotation, rate limiting and proxy trust; current remote read-only gate is insufficient on its own.
7. Remaining moderate npm advisories and long-run disk/DB concurrency stress tests.

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

Do not auto-merge, auto-install Comfy nodes, update CUDA or change Windows Registry. Every next phase must leave a checkpoint and pass CI.