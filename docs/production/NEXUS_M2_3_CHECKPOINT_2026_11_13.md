# Nexus-BTA — M2.3: efficient gallery + inpainting viewport

Checkpoint: **13 November 2026 (America/Sao_Paulo)**.
Branch: `audit/nexus-modernization-m1-2026-11`; draft PR #17. Main remains unchanged.
Prior decisions: `NEXUS_M2_2_CHECKPOINT_2026_11_13.md`, `NEXUS_M2_CHECKPOINT_2026_11_12.md` and `AGENTS.md`.

## Delivered: on-demand image thumbnails

- New `gallery_thumbnail.py` and `GET /api/gallery/thumb/{relative_path:path}?size=320`.
- Images only (PNG/JPEG/WEBP/GIF). JPEG/WEBP image previews are generated on demand with Pillow and encoded as compact WebP without copied EXIF/comment metadata.
- The source cannot exceed **128 MiB** or **64 million pixels**; thumbnail longest edge is constrained to **64–512 px**. Images larger than this may display a broken thumbnail until a separate safe preview is implemented; original content is never silently modified.
- Cached thumbs are keyed by output-root namespace, relative file path, source size, nanosecond mtime and requested dimensions. Cache is written atomically outside `/outputs`, with a **soft 384 MiB** disk budget; periodic oldest-first eviction is amortized to avoid rescanning on every tile.
- Source paths are canonicalized and validated under the output root. Traversal, Windows separators, symlinks to outside, and unsupported extensions are rejected. Cached symlinks are not served. The existing public static outputs route is not expanded.
- New gallery API responses use the thumbnail URL only for images; the active full-resolution preview still uses the original image.
- Video/audio/3D tiles remain lightweight icon placeholders. **No video thumbnail decoding or GLB viewer was added**; those require bounded/sandboxed decoders and runtime QA.

## Delivered: global indexed search, media filter and sort

- `GalleryIndex.page()` accepts query (filename/folder path, <=120 characters), media type (all/image/video/audio/3d), sort (newest/oldest/name/type) and opaque keyset cursor. Sorting/filtering is applied to the **full SQLite index**, not only fetched pages.
- The SQLite schema is upgraded in place to include a sortable filename column; original index entries are preserved. Existing legacy list endpoint `/api/gallery` remains unaffected.
- The API `/api/gallery/page?limit=48&q=...&media=3d&sort=name&cursor=...` validates parameters and passes them to SQL using bind values. Literal `%` and `_` searches are escaped correctly. Cursors carry a query/filter/sort signature to reject reuse under different search settings; old default-view cursors remain accepted.
- React Gallery has a debounced global filename/path search, type dropdown, sorting, loaded/total counters and abortable network queries. Clicking Refresh requests a forced index scan. Search/filters no longer misleadingly operate only on already loaded items.
- Matching model metadata, prompt content and generation labels are **not globally indexed/searchable yet**. SQLite's default LIKE behavior offers primarily ASCII case-insensitivity.

## Delivered: inpaint visual zoom and pan

- `InpaintCanvas.tsx` exposes 100%–400% zoom, pan tool, accessible controls, reset view, pointer capture and pure bounded pan calculations.
- The image and PNG mask canvas are always inside one transformed `inpaint-viewport` to keep visual alignment during editing.
- Mask preview remains capped at 1536px on the longest edge, with previous 32 MiB undo/redo history budget. **This does not yet guarantee source-resolution mask exports for 8K sources**, or visual correctness under all monitors/DPI or resize modes. Those require Windows and ComfyUI/GPU QA.
- Added unit tests for pan bounds and browser acceptance for shared viewport zoom. Browser test uses a 1×1 image fixture; it cannot substitute for a real high-resolution painting session.

## Evidence and CI

- Backend tests: thumbnail generation/cache hit, corrupted source handling, path traversal and symlink rejection, 245+ file pagination, indexed global query/filter/sort, old SQLite schema migration, and HTTP responses.
- Browser smoke (Playwright Chromium, mocked API): Studio/Civitai, presets, Gallery two pages, thumbnail requests, global search, 3D filter and inpaint shared zoom.
- React typecheck/build, Python tests, experimental Rust tests and browser mock acceptance run in separate CI jobs.
- Verified: run `43543037369` passed all four jobs before the final inpaint browser assertion was added. Check the newest run on the PR for the latest commit's result; never assume a green build without checking.

## Safety and limitations before merge

1. Neither the legacy /ui nor the default route is replaced; /app parity and actual local Windows + ComfyUI GPU testing remain incomplete.
2. No remote login/authorization exists for sensitive APIs. CORS and explicit LAN acknowledgement are not authentication. Do not start public tunnels.
3. Thumbnail cache eviction is a soft budget and periodic cleanup; complete absence of memory or file-handle leaks is not yet proven by stress testing.
4. SQLite file rescans are periodic, not event-driven; first index pass is O(number of files). Multi-worker/simultaneous write concurrency needs stress validation.
5. GLB interactive preview, video thumbnail capture, full resolution mask alignment, audio visualizer, global prompt/model metadata search and real workflow compatibility are still pending.
6. ComfyUI, custom nodes, NVIDIA/Torch, Windows registry and user model paths are intentionally untouched.

## Next priority: M2.4

1. Safe opt-in preview pipeline for local videos and GLB/GLTF, with size bounds and lifecycle cleanup; evaluate native Three.js viewer vs external/web component only with lockfile/reproducible dependency tests.
2. Global indexed search over *bounded* metadata and a scalable thumbnail invalidation policy; preserve app privacy.
3. Runtime `object_info`/Comfy capability adapter, missing-node diagnostics and per-workflow compatibility matrix without automatic installation.
4. High-resolution inpaint/source-to-mask scaling test matrix, pan/brush long-session profiling and screen zoom/DPI QA.
5. Begin backend `main.py` and `workflows.py` decomposition via routers/services with unchanged wire protocol.

## Local branch continuation

```powershell
git fetch origin
git switch audit/nexus-modernization-m1-2026-11
git pull --ff-only
cd frontend
npm ci
npm run check
npm run test:unit
# Browser smoke: install Playwright Chromium; then npm run dev and npm run smoke:m2.
# No LAN/tunnel needed.
```

Read `AGENTS.md` and this checkpoint before next automation/Codex handoff. Every material milestone needs a commit and a reproducible gate.
