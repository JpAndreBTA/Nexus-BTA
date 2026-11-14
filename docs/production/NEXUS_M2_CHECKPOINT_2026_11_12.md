# Nexus-BTA — M2 UI modernization checkpoint (12 November 2026, Brazil)

Source: `audit/nexus-modernization-m1-2026-11` (draft PR #17).
Baseline: main `89e8da08bb115fc2cbaa89635837c99ea50b05ca`.
**Do not merge to main until actual Windows + ComfyUI/GPU/legacy acceptance is complete.**

## Completed and committed

- Split React Studio workflow preview (`StudioWorkflowGraph.tsx`), Ideogram regional editor (`IdeogramRegionEditor.tsx`), and model preset helper (`StudioCatalog.ts`). `HomePage.tsx` decreased from ~2052 lines to ~1564 lines; core Studio business orchestration is still monolithic.
- Extracted AppTopbar from the route tree, and Civitai into a lazy-loaded modal with its own search, download polling, focus, keyboard and scroll lifecycle. React router now ~124 lines; the legacy 2.45 MB `index.html` remains unchanged.
- Inpaint editor changed from a fixed 768px max surface to a bounded 1536px max, retains the legacy black/white mask PNG contract, uses reusable export canvas, bounded undo/redo history (32 MiB approximate RGBA snapshot budget), tracked pointer capture and reset on same-dimension reference replacement.
- Pure mask geometry, history and alpha threshold functions in `maskPixels.ts` have browser-independent unit tests. Experimental Rust mask crate remains opt-in, unconnected to the browser workflow.
- Added user-adjustable Studio control and gallery panel widths for desktop; the existing responsive mobile/tablet layout wins at smaller viewports.
- Appearance/layout JSON v1 export and import with strict allowlisted fields, 16 KB size bound and tests; no prompts, paths, credentials or model files are transferred in this preset.
- Dependency audit located vulnerable `seroval@1.5.4` in the TanStack Router tree; locked and overridden to verified `1.6.8` using npm published SHA-512 integrity. CI now enforces `npm audit --audit-level=high`.
- CI now has four gates: Python security/gallery tests, Rust mask tests, React typecheck/build + unit tests + npm audit, and a Chromium browser smoke (Ubuntu 24.04, mocked backend). All four jobs passed at run `43352756989`.

## Explicit limits and unverified areas

1. Browser smoke uses an API mock. It verifies visible SPA navigation, Civitai mock search, modal keyboard close, theme persistence, layout JSON import/export, gallery empty state, absence of uncaught JS exceptions. It does **not** execute ComfyUI, load local custom nodes/models, perform a real Civitai download, measure generation, test user Windows or compare visual screenshots.
2. Inpaint canvas alignment, denoise, undo/redo under long real drawing sessions, real backend mask resampling and VRAM/heap memory need hands-on browser/GPU QA.
3. React `/app` is **not** feature-complete with legacy `/ui`; never switch the default route until parity is demonstrated.
4. Network launchers require acknowledgement, but there is no authenticated server gateway or authorization for sensitive API/file routes. LAN/public use is not homologated.
5. `npm ci` reported four moderate advisories after fixing the high/critical `seroval` advisory. They still require separate triage; do not claim a zero-vulnerability dependency tree.
6. The 3D route and legacy workflow migration remain experimental. Keep all existing templates, models, user settings and local runtime intact.

## Next production unit (M2.2)

1. Extract Studio generation orchestration into hooks/services, retaining request payload parity for txt2img, img2img, inpaint/outpaint, ControlNet, video, Director, MiniMax and 3D.
2. Build real browser E2E against the actual FastAPI backend without GPU (contract fixture), then GPU-run matrix for production templates.
3. Upgrade Gallery from bounded scan of 200 files to a SQLite-indexed paginated API with video/image thumbnails, 3D GLB previews, file watching and safe path isolation.
4. Improve inpaint 1:1 source-to-mask alignment, zoom/pan, variable-res previews and memory profiling with long-session leaks.
5. Modular accessibility for all modals, Esc/focus/scroll cleanup, object URL lifecycle and abortable requests.
6. Capability-based Comfy adapter: validate `object_info`, workflow imports, missing nodes and model-specific settings. No automatic update of ComfyUI/custom nodes without compatibility gate.

## Continuation commands

```powershell
git fetch origin
git switch audit/nexus-modernization-m1-2026-11
git pull --ff-only
cd frontend
npm ci
npm run check
npm run test:unit
# Browser smoke needs a dev server and installed Chromium.
# NEVER invoke network tunnels for browser testing.
```

Read `AGENTS.md`, `frontend/MIGRATION.md`, `docs/audit/NEXUS_UI_AUDIT_2026_11_12.md` and this checkpoint before continuing. Preserve separate projects and avoid unnecessary re-reading of the legacy HTML monolith.
