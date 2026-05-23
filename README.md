# Nexus BTA

Nexus BTA is a local AI image and video studio built around an embedded ComfyUI runtime. It gives you one focused interface for SD 1.5, SDXL, Flux, Qwen, Lumina, WAN 2.2, LTX 2.3, Anima, LoRAs, ControlNet, inpaint, img2img, txt2img, and short video workflows.

New here? Start with `run.bat`, open the UI, pick a model tab, write a simple prompt, and generate. Nexus is designed to keep the heavy ComfyUI wiring available when you need it, while keeping the everyday controls close at hand.

## First Look

The `examples/` folder includes quick visual references for the main workspaces.

![Nexus BTA main interface](examples/Ui_layout.png)

The main layout keeps model presets, generation controls, the viewer, workflow tabs, and the gallery in one focused workspace.

![Inpaint workspace](examples/Inpaint_layout.png)

Use the inpaint canvas to load a reference, paint masks, undo/redo brush edits, and send gallery images directly into the canvas.

![Node workflow workspace](examples/Node_Workflow_layout.png)

Switch to Node Workflow when you want to inspect or tune the generated Comfy graph without leaving Nexus.

![Civitai browser modal](examples/Civitai_Modal.png)

The Civitai modal helps browse models, download assets, and route them into the right Nexus model folders.

## Why It Feels Fast

- One launcher: `run.bat` starts Nexus and the embedded ComfyUI runtime.
- Smart model folders: checkpoints, UNET/diffusion models, VAEs, text encoders, LoRAs, ControlNet, and video assets are discovered automatically.
- Template-aware side menu: preset changes keep workflow nodes, CFG, steps, resolution, video motion, LoRA stacks, and ControlNet in sync.
- Template-scoped references: Qwen multi-reference image slots stay on Qwen, while SD/SDXL/Flux/Lumina/Anima/WAN/LTX return to their own img2img or img2video source controls.
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

## Model Assets

Use [requirements/model_assets.md](requirements/model_assets.md) as the friendly download map for base models, lightweight LoRAs, distilled LoRAs, LTX 2.3, WAN 2.2, Flux, Anima and textual inversion embeddings.

- LoRAs live in `models/loras` and can be grouped by family, such as `sd15`, `sdxl`, `flux`, `anima`, `wan` and `ltx`.
- Textual inversion files live in `models/embeddings`; the prompt Emb buttons insert them into positive or negative prompts for SD 1.5, SDXL, Pony and Illustrious routes.
- WAN 2.2 and LTX 2.3 use video-specific encoders, so use their model, VAE, text encoder and distilled LoRA assets instead of classic textual inversion. WAN 2.2 4-step runs need the matching high/low 4-step LoRA pair under `models/loras/wan`.
- LTX 2.3 assets are available from [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3); WAN 2.2 assets are available from [Wan-AI](https://huggingface.co/Wan-AI).

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
