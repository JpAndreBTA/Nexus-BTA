# Nexus-BTA — M2.2: Studio request contracts & indexed Gallery (13 Nov 2026)

## Reference and safety

- Branch: `audit/nexus-modernization-m1-2026-11`, draft PR #17.
- Main baseline remains unchanged and `/ui` remains available.
- This checkpoint extends `NEXUS_M2_CHECKPOINT_2026_11_12.md` and `NEXUS_UI_AUDIT_2026_11_12.md`.
- No automatic ComfyUI or custom-node upgrades, no touching user model folders, no Windows registry changes and no public network enablement.

## Shipped — M2.2a: generation service and contract matrix

- New `frontend/src/views/home/generationPayload.ts`: pure serializer from Zustand state to the existing FastAPI `GenerateRequest`. Preserves SD txt2img, img2img/inpaint, ControlNet, Wan video, LTX Director, Ideogram 4 regional layout and MiniMax Music 3.
- Centralized preflight checks with the same error messages in `validateGeneration`.
- Request snapshots are now passed directly to the React Query mutation instead of dispatching from a stale zero-argument closure.
- Fixed a missing React memo dependency: changing the detected Music3 hardware profile now updates the request's runtime precision and VRAM policy.
- The `HomePage.tsx` file is now about 1,416 lines (it was ~2,052 before M2 began). **The generation UI itself is not fully modular yet**.
- Unit tests `generationPayload.test.mjs` cover the contract variants, invalid input and the Music3 profile change. No actual ComfyUI generation is claimed.

## Shipped — M2.2b: indexed gallery and browser acceptance

- `backend/nexus_backend/gallery_index.py`: persistent SQLite index using only relative paths, file mtime, size and type; keyset cursor paging with deterministic ordering and bounded result sizes. Index database sits in the current user's application cache outside `/outputs`.
- Batches update metadata, remove missing entries and skip video-sidecar images. Symlink paths outside the permitted root are ignored; invalid cursors and page sizes are rejected.
- Gallery scans are rate-limited to roughly every 10 seconds per process. This is a periodic directory scan, **not** a file watcher, and initial indexing remains O(files).
- `backend/nexus_backend/gallery_api.py`: separate, independently testable FastAPI router. `GET /api/gallery/page?limit=48&cursor=...` responds with `{items, total, next_cursor}`; optional `refresh=true` forces a rescan. Existing `/api/gallery` returns the legacy list of up to 200 unchanged.
- New paginated response does not expose the absolute path of the Windows filesystem. Each indexed path is checked again before metadata is read. SQLite connections are closed explicitly and metadata extraction is offloaded off the FastAPI event loop.
- `frontend/src/views/gallery/GalleryPage.tsx` now requests cursor pages as required, with explicit `loaded / total`, load-more action, bounded media previews and abortable requests.
- Search and sort on the React client currently operate only on *loaded* pages, not the entire index. UI labels and documentation state this limitation to avoid claiming global search parity.
- Tests: SQLite 245-media pagination, refresh/deletion, bad cursors, symlink escape and database isolation. FastAPI/HTTP contract test with 120 media records validates page composition, cursor and no absolute-path leak.
- Chromium browser smoke on mocked backend tests Studio, Civitai, appearance import/export, Gallery empty state and two-page Gallery loading. It uploads `nexus-m2-browser-gallery` screenshot artifact for short-term visual QA.
- Py, frontend, Rust and browser jobs passed at run `43536356408` after the browser's selector was corrected. This checkpoint also includes follow-up off-main code changes; verify the latest run before merging.

## Known limitations / QA gates

1. **Real Windows + ComfyUI/GPU smoke is still required**, with SD/SDXL/Flux/Qwen/Ideogram4/WAN/LTX/Director/Music3 and model asset downloads. Browser and FastAPI tests mock or isolate external execution.
2. Real inpainting still needs brush/stroke, 1:1 mask/source alignment, zoom/pan and memory/VRAM profiling on large user images.
3. True image/video thumbnails and GLB/3D preview renderer are not implemented in this milestone. Loading raw full-resolution image for gallery preview can still cost memory.
4. Per-directory watcher/notifications, full-text indexed metadata search, server-side sorts and thumbnail caching are pending M2.3.
5. LAN/tunnel/API still lacks complete authentication and authorization. Host acknowledgement and local CORS checks are NOT an access-control layer.
6. UI `/app` is not yet feature-complete with `/ui`; do not switch default route. Four npm moderate advisories remain separately triaged (CI blocks high/critical).
7. There is no automatic GitHub merge. PR #17 is draft until visual, Windows, security and workflow parity reviews are complete.

## M2.3 recommended next work

1. Add safe, disk-cached bounded thumbnails for images and videos; explicit CPU and input-size budgets, no eager media decoding.
2. Server-side search, filter, and ordering using SQLite indexed metadata, with cursor consistency under rescans and multi-worker synchronization.
3. Optional sandboxed 3D preview (GLB first) loaded as a separate React chunk.
4. Source-to-mask full-res mapping, inpaint zoom and pan, and long-lived brush session tests.
5. Build a Comfy capability/version adapter with `object_info` and missing-node diagnostics; gate upgrades via compatibility tests, no unapproved installations.
6. Split remaining Studio UI by operation (image, video, director, audio, inpaint) and disentangle the 600 KB backend `main.py`.

## Local acceptance after syncing branch

```powershell
git fetch origin
git switch audit/nexus-modernization-m1-2026-11
git pull --ff-only
cd frontend
npm ci
npm run test:unit
npm run check
# To run Chromium mock smoke locally, install Playwright Chromium and start npm run dev.
# Do not start LAN/tunnel modes just for smoke testing.
```

Follow `AGENTS.md`; keep this checkpoint for the next Codex worker.
