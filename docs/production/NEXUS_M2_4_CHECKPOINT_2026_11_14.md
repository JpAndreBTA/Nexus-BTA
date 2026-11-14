# Nexus-BTA — M2.4: Comfy capabilities, video posters and GLB preview

**Date:** 2026-11-14 (America/Sao_Paulo). Working branch: audit/nexus-modernization-m1-2026-11. Draft PR #17. Main stays at the original baseline; /ui stays available.

Follow M2.3 checkpoint and the architecture audit before modifying code.

## 1. Bounded on-demand video posters

- backend/nexus_backend/gallery_video.py; new GET /api/gallery/video-thumb/{relative_path:path}?size=320 endpoint in gallery_api.py.
- Existing local FFmpeg or the installed imageio_ffmpeg executable only; no package installation, network download or shell execution. FFmpeg has file-only input protocols.
- One frame; decoder timeout 12 seconds; source <=2 GiB; poster <=512 px; temporary decoded PNG <=16 MiB; output WebP without copied metadata.
- Private cache outside /outputs, canonical-root path checks, key by relative path/mtime_ns/size, atomic output and best-effort soft 384 MiB eviction.
- React Gallery renders posters only when available. A missing decoder or decoding failure falls back to a lightweight video icon; audio remains an icon.
- Python regression tests cover flags, mocked video decode, caching, errors and HTTP route. Chromium acceptance checks the video poster request.
- The first frame may be black or unrepresentative; codecs, very large clips, or Windows FFmpeg paths require real-device QA.

## 2. Read-only ComfyUI node registration diagnostics

- backend/nexus_backend/comfy_capabilities.py; GET /api/comfy/capabilities?classes=ClassA,ClassB.
- Probes only an existing running ComfyUI. Does NOT call ensure_running, install dependencies, update custom nodes, modify configs or restart ComfyUI.
- Limits registry body to 16 MiB and 64 requested classes; returns only the registration summary, not full object_info.
- Offline/unavailable is explicitly unverified, not misreported as missing nodes. Only a successful registry read can classify nodes as absent.
- Studio Node Workflow view checks class types from the real backend visual graph; editor-generated diagnostic nodes are excluded.
- HTTP unit tests verify registered/missing/offline and malformed inputs, and Chromium checks offline UX. Node registration alone does not prove workflow execution, VRAM or model files.

## 3. Experimental lazy GLB 3D geometry preview

- glbPreview.ts parses bounded GLB v2 (up to 24 MiB, first embedded TRIANGLES primitive, up to 65,536 triangles), validates chunk/accessor/index bounds and refuses external glTF buffers.
- GalleryGlbPreview.tsx loads lazily for selected .glb only, renders a flat-shaded WebGL2 mesh and supports mouse orbit, wheel zoom, keyboard rotation and Home reset.
- Fetch is streamed with an abort signal and hard byte cap; GPU buffers, listeners and resize observers are released when the user navigates away.
- Headless Chromium loads a genuine small GLB triangle fixture and generates the gallery-glb-preview.png screenshot artifact; pure parser tests cover valid/malformed GLB.
- **Deliberate limitations:** no PBR, hand-painted textures, multiple mesh primitives, node hierarchy transforms, skeletons, animation, lights, external resources or GLTF .gltf support. This is a limited first-mesh geometry QA view, not a full asset viewer. Unsupported files keep the original download action.

## Evidence and quality gate

- Verified code SHA: 5324ada92b46d4282e2f468aca4e15ae24a30b4e.
- GitHub Actions run 43552213981: 4/4 successful (Python backend, React/TS+high npm audit and unit, Rust test, real Chromium with mocked API).
- Browser artifact nexus-m2-browser-gallery includes paginated gallery and GLB preview PNG screenshots (seven-day retention).
- No claim is made of Windows local ComfyUI/GPU acceptance, production model generation, every 3D asset, all video codecs, memory leak absence or safe public server exposure.
- Four moderate npm advisories remain pending triage, high/critical audit gate passes.

## Next M2.5 priorities

1. Credential-based, permission-limited gateway before any LAN/tunnel/public usage; CORS is not authentication.
2. Real Windows/ComfyUI/GPU acceptance matrix (SD/XL, WAN, LTX Director, Music3, LoRAs, Civitai, image/video/audio/3D generation), including custom node ABI, imports and model files.
3. Optional full scene/PBR GLB viewer with strict size, fetch and resource limits. Current preview is intentionally geometry-only.
4. True source-resolution inpaint mask mapping with long-session memory profiling and additional brush/pan acceptance.
5. More targeted breakdown of main.py, workflows.py and Studio features into services/routers without breaking legacy API.
6. Metadata search, disk cache stress and file watching with concurrency support.

## Continuation

Read AGENTS.md, this M2.4 checkpoint, M2.3, M2.2 and frontend/MIGRATION.md. Do not auto-merge PR #17, do not change /ui default, do not install ComfyUI upgrades or touch models without compatibility gates.
