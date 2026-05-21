# Nexus BTA

Nexus BTA is a local image and video generation workbench with an embedded ComfyUI runtime, a Nexus-designed UI, template-aware model folders, visual workflows, LoRA support, Civitai integration, and SD 1.5 / SDXL / Flux / Qwen / Lumina / WAN 2.2 / LTX 2.3 / Anima oriented presets.

## Highlights

- Embedded backend and ComfyUI runtime launched from `run.bat`.
- Template-specific model folders such as `models/checkpoints/anima`, `models/checkpoints/ltx`, and `models/checkpoints/sdxl`.
- txt2img, img2img/img2vid, inpaint, inpaint sketch, extras, PNG info, checkpoint merge, settings, extensions, gallery, assets, and node workflow views.
- Live Gallery sync watches `output/` through the backend, refreshes while the app is open, and updates after generation without a browser refresh.
- Visual workflow tabs with editable nodes, notes, pins, connect/disconnect, bypass/enable/disable, and side-menu synchronization.
- Optional multi-LoRA concept stacks from the side menu, with unified preview cards, auto-apply on modal backdrop close, per-LoRA strengths, and LTX 2.3 LoRA routing.
- ControlNet panel for SD 1.5 and SDXL image workflows with enable/disable, Canny/OpenPose/Depth/Lineart/Tile type selection, model selection, control image drop zone, strength, start/end range, and side-menu-to-workflow synchronization. Canny ControlNet models are included in the expected `models/controlnet` layout.
- LTX 2.3 img2vid support with 512x512, 4 seconds, 24 FPS, Video VAE, Audio VAE, active audio toggle, and latent upscale selection.
- WAN 2.2 high/low noise video support with 512x512, 2 seconds, 24 FPS, Euler/Simple, CFG 1, and 4-step split sampling.
- Civitai browsing, account token support, URL resolution, and downloads to the correct Nexus folders.
- Outputs are written to `output/`; runtime, model weights, third-party workflows, and generated media are ignored by git.

## Quick Start

1. Place or bootstrap the embedded runtime:
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts/bootstrap_nexus_runtime.ps1 -CopyPythonEnv
   ```

2. Add compatible models under the template folders:
   ```text
   models/checkpoints/sd15
   models/checkpoints/sdxl
   models/checkpoints/qwen
   models/checkpoints/ltx
   models/checkpoints/anima
   models/unet/wan
   models/loras
   models/vae
   models/text_encoders
   models/controlnet
   models/upscale_models
   models/latent_upscale_models
   ```

3. Start Nexus:
   ```bat
   run.bat
   ```

The launcher waits for the backend, starts embedded ComfyUI on demand, opens `http://127.0.0.1:7861/ui`, and stops backend/runtime services when the launcher window is closed.

## Updates

Run:

```bat
update.bat
```

This checks git updates, verifies runtime dependencies, and recreates expected folder structure.

## Workflows

Nexus-owned base workflows live in `workflows/nexus_base`. Third-party workflows imported into `workflows/comfyui` are intentionally ignored by git.

Loaded workflows are checked against the embedded ComfyUI object registry. If custom nodes are missing, Nexus Manager suggests or installs compatible dependencies when available.

## Recommended Smoke Battery

Use `run.bat`, open `http://127.0.0.1:7861/ui`, and verify:

- SD 1.5, SDXL, Qwen, Anima: txt2img, img2img, and inpaint paths.
- SD 1.5 and SDXL: ControlNet Canny at 512x512 with a loaded control image and compatible `models/controlnet` model.
- LTX 2.3: img2vid at 512x512, 4 seconds, 24 FPS, Euler CFG++ / Quadratic, 8 steps, CFG 1.
- WAN 2.2: I2V/T2V at 512x512, 2 seconds, 24 FPS, Euler / Simple, 4 steps, CFG 1.
- Gallery output metadata: prompt, model, seed, sampler, scheduler, steps, CFG, dimensions, preset, and workflow id.

## Latest Verification

May 21, 2026 smoke results:

- SD 1.5 ControlNet Canny: completed at 512x512, non-black output.
- SDXL ControlNet Canny: completed at 512x512, non-black output.
- Qwen sentinel: completed at 512x512 with CFG 1 and 4 steps, non-black output.
- WAN 2.2: completed at 512x512, 24 FPS, 49 frames, about 2.04 seconds.
- LTX 2.3: completed at 512x512, 24 FPS, 97 frames, about 4.04 seconds, AAC audio present.

## License

MIT. See [LICENSE](LICENSE).
