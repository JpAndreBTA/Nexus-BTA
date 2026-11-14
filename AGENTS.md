# AGENTS.md — Nexus-BTA

This repository is the independent **Nexus-BTA** local generative AI studio, not VEYLUM or Game3.

1. Read `docs/production/NEXUS_M2_7_CHECKPOINT_2026_11_14.md`, `docs/production/NEXUS_M2_6_CHECKPOINT_2026_11_14.md`, `docs/production/NEXUS_M2_5_CHECKPOINT_2026_11_14.md`, `docs/production/NEXUS_M2_4_CHECKPOINT_2026_11_14.md`, `docs/production/NEXUS_M2_3_CHECKPOINT_2026_11_13.md`, `docs/production/NEXUS_M2_2_CHECKPOINT_2026_11_13.md`, `docs/production/NEXUS_M2_CHECKPOINT_2026_11_12.md`, `docs/audit/NEXUS_UI_AUDIT_2026_11_12.md` and `frontend/MIGRATION.md` before major work. Current release remains `/ui` while React `/app` migrates by bounded modules.
2. Keep all custom user workflow JSON, installed models, runtime paths, user layouts, local presets, Civitai settings, generation parameters and legacy fallback working. Do not overwrite user/runtime folders or stage binary model assets.
3. Work on feature branches with coherent milestones. No automatic merge to main. Use a meaningful gate per milestone: frontend typecheck/build, Python contract tests, Rust tests if touched, then browser+GPU visual QA where available. Clearly distinguish static evidence from runtime-tested behavior.
4. Preserve Python/ComfyUI/PyTorch for generation; introduce Rust/WASM incrementally only for benchmarked CPU-bound, local desktop or image-processing work. The experimental `crates/nexus-image-core` is not integrated yet.
5. Do not expose `/inputs`, `/model-assets`, installation, process or config endpoints to public networks. CORS is not authentication. Never use StartTunnel as a substitute for an authenticated gateway.
6. Prefer small React features and backend routers over further growing `HomePage.tsx`, `index.html`, `main.py` or `workflows.py`. Keep API/workflow schemas backward compatible, handle unknown nodes losslessly, and maintain rollback.
7. Record decisions, evidence, regressions, and accepted milestones in docs. Avoid claim of a finished migration without feature parity and real Windows/GPU testing.
