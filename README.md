# Nexus BTA

Nexus BTA is a local AI image and video studio built around an embedded ComfyUI runtime. It gives you one focused interface for SD 1.5, SDXL, Flux, Qwen, Lumina, WAN 2.2, LTX 2.3, Anima, LoRAs, ControlNet, inpaint, img2img, txt2img, and short video workflows.

## Why It Feels Fast

- One launcher: `run.bat` starts Nexus and the embedded ComfyUI runtime.
- Smart model folders: checkpoints, UNET/diffusion models, VAEs, text encoders, LoRAs, ControlNet, and video assets are discovered automatically.
- Template-aware side menu: preset changes keep workflow nodes, CFG, steps, resolution, video motion, LoRA stacks, and ControlNet in sync.
- Clean gallery: final outputs refresh on launch, when the gallery opens, and after a new generation. Temporary Comfy previews are ignored.
- Visual workflow view: inspect and tune nodes without leaving the app.
- Civitai and Concept LoRA modals: browse, download, preview, multi-select, and route assets into the right local folders.

## Start

```bat
run.bat
```

Then open:

```text
http://127.0.0.1:7861/ui
```

If this is a fresh machine, bootstrap the runtime first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_nexus_runtime.ps1 -CopyPythonEnv
```

## Model Folders

Keep models under `models/`:

```text
models/checkpoints
models/unet
models/diffusion_models
models/loras
models/vae
models/text_encoders
models/controlnet
models/upscale_models
models/latent_upscale_models
```

UNET-style models can live in either `models/unet` or `models/checkpoints`; Nexus resolves both.

## Verified Smoke Battery

Last verified on May 21, 2026:

- SD 1.5 and SDXL image generation, img2img, inpaint, and ControlNet Canny.
- Qwen at 512x512 with CFG 1 and 4 steps.
- WAN 2.2 at 512x512, 2 seconds, 24 FPS.
- LTX 2.3 at 512x512, 4 seconds, 24 FPS, with audio present.
- Anima with Concept LoRA selection and gallery metadata.

## Notes

- Generated media stays in `output/`.
- Temporary inputs, masks, Comfy previews, and Nexus temp files are cleaned after generation.
- Runtime files, model weights, generated media, and local settings are ignored by git.

## License

MIT. See [LICENSE](LICENSE).
