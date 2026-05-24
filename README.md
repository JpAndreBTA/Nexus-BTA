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

Switch to Node Workflow when you want to inspect or tune the generated Comfy graph without leaving Nexus. The editor supports a Blender-style add-node menu with search, categorized node presets, click-to-connect ports, click-to-unlink inputs, Ctrl+click multi-select, and drag-box selection. When this tab is active, Nexus sends the edited visual graph as the Comfy workflow override instead of treating the view as a mockup.

![LTX 2.3 linear video workspace](examples/LTX_2.3.png)

The LTX 2.3 Linear View keeps img2video/txt2video controls synchronized with the Comfy workflow, including video VAE, audio VAE, distilled LoRAs and latent upscaling.

![LTX 2.3 Director timeline](examples/LTX_2.3Director.png)

The LTX 2.3 Director Suite adds a timeline for image, text, video and audio segments, with per-segment prompts, negative prompts, crop/camera controls, custom audio and generated speech/ambience routing.

![Civitai browser modal](examples/Civitai_Modal.png)

The Civitai modal helps browse models, download assets, and route them into the right Nexus model folders.

## Why It Feels Fast

- One launcher: `run.bat` starts Nexus and the embedded ComfyUI runtime.
- Smart model folders: checkpoints, UNET/diffusion models, VAEs, text encoders, LoRAs, ControlNet, and video assets are discovered automatically.
- Template-aware side menu: preset changes keep workflow nodes, CFG, steps, resolution, video motion, LoRA stacks, and ControlNet in sync.
- Template-scoped references: Qwen multi-reference image slots stay on Qwen, while SD/SDXL/Flux/Lumina/Anima/WAN/LTX return to their own img2img or img2video source controls.
- Clean gallery: final outputs refresh on launch, when the gallery opens, and after a new generation. Temporary Comfy previews are ignored.
- Organized outputs: image saves land under `output/image/` and video saves land under `output/video/`, with date/time prefixes for easier sorting.
- Visual workflow view: inspect, link, multi-select, move and tune nodes without leaving the app; edited graphs are sent through the backend as Comfy workflow overrides.
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

`run.bat`, `update.bat`, and the bootstrap script also check LTX Director dependencies. They install the WhatDreamsCost Director node, Kijai's Mel-Band RoFormer node, its Python requirements, and the `MelBandRoformer_fp16.safetensors` model into the expected local folders when they are missing.

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
models/frame_interpolation
models/background_removal
```

UNET-style models can live in either `models/unet` or `models/checkpoints`; Nexus resolves both.

## Model Assets

Use [requirements/model_assets.md](requirements/model_assets.md) as the friendly download map for base models, lightweight LoRAs, distilled LoRAs, LTX 2.3, WAN 2.2, Flux, Anima and textual inversion embeddings.

- LoRAs live in `models/loras` and can be grouped by family, such as `sd15`, `sdxl`, `flux`, `anima`, `wan` and `ltx`.
- Textual inversion files live in `models/embeddings`; the prompt Emb buttons insert them into positive or negative prompts for SD 1.5, SDXL, Pony and Illustrious routes.
- WAN 2.2 and LTX 2.3 use video-specific encoders, so use their model, VAE, text encoder and distilled LoRA assets instead of classic textual inversion. WAN 2.2 4-step runs need the matching high/low 4-step LoRA pair under `models/loras/wan`.
- LTX 2.3 assets are available from [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3); WAN 2.2 assets are available from [Wan-AI](https://huggingface.co/Wan-AI).
- LTX 2.3 latent upscale is part of the normal route. For a `512x512` output, Nexus samples the base latent at `256x256`, applies `LatentUpscaleModelLoader` + `LTXVLatentUpsampler`, refines it, and then decodes the final video.
- Extras uses `models/upscale_models` for image/video raster upscalers, `models/frame_interpolation` for RIFE/FILM frame interpolation, and `models/background_removal` for 2026 background-removal routes such as ComfyUI native BiRefNet. Alpha-aware video export is exposed as PNG sequence or MOV ProRes 4444.
- Remove BG expects `models/background_removal/birefnet.safetensors` for the native ComfyUI BiRefNet workflow. RMBG-2.0, InSPyReNet and BEN/BEN2 remain selectable compatibility routes when matching ComfyUI custom nodes are installed.

## Output Layout

Generated media is grouped by type:

```text
output/image/YYYYMMDD_HHMMSS_<preset>_<activity>_...
output/video/YYYYMMDD_HHMMSS_<preset>_<activity>/YYYYMMDD_HHMMSS_<preset>_<activity>_...
```

Nexus applies this naming layer to default templates, loaded workflows and visual workflow overrides before sending the job to ComfyUI. Video routes, including image-sequence exports, receive a per-generation dated folder so PNG sequences do not mix with other runs. The Gallery can navigate output folders with visible folder cards, back/forward history, optional folder picker and date/type sorting.

## Verified Smoke Battery

Last verified on May 24, 2026:

- SD 1.5 and SDXL image generation, img2img, inpaint, and ControlNet Canny.
- Qwen at 512x512 with CFG 1 and 4 steps.
- WAN 2.2 at 512x512, 2 seconds, 24 FPS.
- LTX 2.3 at 512x512, 4-5 seconds, 24 FPS, with latent upscale/refiner routed in both Linear View and Director Suite.
- Extras image upscale with Remacri/UltraSharp/RealESRGAN, RIFE `rife_v4.26`, video upscale, PNG sequence alpha and MOV ProRes 4444 alpha export.
- Extras Remove BG frontend plan sync with `models/background_removal/birefnet.safetensors`, recommended `Remove BG Image` and `Remove BG Video` presets, video PNG sequence or MOV ProRes 4444 alpha export, RGBA/mask output options and folder-aware Gallery navigation.
- LTX 2.3 Director Suite with custom background audio, generated speech/ambience segments, per-segment negative prompts and non-black video output.
- Anima with Concept LoRA selection and gallery metadata.
- Node Workflow editor smoke: menu search, categorized add-node menu, visual multi-selection, grouped drag behavior, port connection/unlink affordances, and workflow override routing from the active graph.

## Notes

- Generated media stays in `output/image` and `output/video`.
- Temporary inputs, masks, Comfy previews, and Nexus temp files are cleaned after generation.
- Runtime files, model weights, generated media, and local settings are ignored by git.

## License

MIT. See [LICENSE](LICENSE).
