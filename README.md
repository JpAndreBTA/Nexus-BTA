# Nexus BTA

Nexus BTA is a local AI image and video studio built around an embedded ComfyUI runtime. It gives you one focused interface for SD 1.5, SDXL, Flux, Qwen, Lumina, WAN 2.2, LTX 2.3, Anima, LoRAs, ControlNet, inpaint, img2img, txt2img, and short video workflows.

New here on Windows? Start with `run.bat`, open the UI, pick a model tab, write a simple prompt, and generate. Linux and macOS can run the backend/UI manually, but the bundled one-click launcher and dependency installers are Windows PowerShell scripts today. Nexus is designed to keep the heavy ComfyUI wiring available when you need it, while keeping the everyday controls close at hand.

## First Look

<video src="https://i.imgur.com/v9pnHrc.mp4" controls muted loop playsinline width="100%"></video>

The `examples/` folder includes quick visual references for the main workspaces.

![Nexus BTA main interface](examples/Ui_layout.png)

The main layout keeps model presets, generation controls, the viewer, workflow tabs, and the gallery in one focused workspace.

![Inpaint workspace](examples/Inpaint_layout.png)

Use the inpaint canvas to load a reference, paint masks, undo/redo brush edits, and send gallery images directly into the canvas.

![Node workflow workspace](examples/Node_Workflow_layout.png)

Switch to Node Workflow when you want to inspect or tune the generated Comfy graph without leaving Nexus. The editor supports a Blender-style add-node menu with search, categorized node presets, click-to-connect ports, click-to-unlink inputs, Ctrl+click multi-select, and drag-box selection. When this tab is active, Nexus sends the edited visual graph as the Comfy workflow override instead of treating the view as a mockup.

![LTX 2.3 linear video workspace](examples/LTX_2.3.png)

The LTX 2.3 Linear View keeps img2video/txt2video controls synchronized with the Comfy workflow, including video VAE, audio VAE, distilled LoRAs, latent upscaling, IC-LoRA identity/detailing, Motion Transfer and the optional Transition LoRA for start/end frame motion.

![LTX 2.3 Director timeline](examples/LTX_2.3Director.png)

The LTX 2.3 Director Suite adds a timeline for image, text, video and audio segments, with per-segment prompts, negative prompts, crop/camera controls, custom audio, generated speech/ambience routing, Motion Transfer segments, Transition LoRA end frames and joined final video export.

![Civitai browser modal](examples/Civitai_Modal.png)

The Civitai modal helps browse models, download assets, and route them into the right Nexus model folders.

## Why It Feels Fast

- One launcher on Windows: `run.bat` starts Nexus and the embedded ComfyUI runtime.
- Smart model folders: checkpoints, UNET/diffusion models, VAEs, text encoders, LoRAs, ControlNet, and video assets are discovered automatically.
- Template-aware side menu: preset changes keep workflow nodes, CFG, steps, resolution, video motion, LoRA stacks, and ControlNet in sync.
- LTX 2.3 Motion Transfer: Pose, Canny, Depth and Camera/Cameraman modes use IC-LoRA-compatible workflows with target identity conditioning, latent upscale x2, optional IC Detailer and Director segment rendering.
- Modern ControlNet routes: Flux, Qwen and Z-Image/ZImage presets expose compatible ControlNet model selection from the side menu and Civitai model browser.
- Inpaint workspace: LanPaint is the default inpaint template, Differential Diffusion remains selectable, and the canvas includes paint/remove masks, outpaint expand canvas, magic wand/select object, undo and redo.
- Extras video tools: video upscale can route through classic upscalers, FlashVSR-ready and SeedVR2-ready engines, LTX IC Detailer refine/upscale, interpolation, denoise, face restoration and MP4 encode.
- Template-scoped references: Qwen multi-reference image slots stay on Qwen, while SD/SDXL/Flux/Lumina/Anima/WAN/LTX return to their own img2img or img2video source controls.
- Clean gallery: final outputs refresh on launch, when the gallery opens, and after a new generation. Temporary Comfy previews are ignored.
- Organized outputs: image saves land under `output/image/`, normal video saves land under `output/video/`, Director joined videos land under `output/videos/`, and Director segment renders remain archived under `output/director/<date_time>/segments/`.
- Visual workflow view: inspect, link, multi-select, move and tune nodes without leaving the app; edited graphs are sent through the backend as Comfy workflow overrides.
- Civitai and Concept LoRA modals: browse, download, preview, multi-select, and route assets into the right local folders.
- Train LoRA workspace: prepare SD/SDXL/Flux/Qwen/WAN/LTX/Anima LoRA jobs, including LTX 2.3 Motion LoRA, Audio-Video LoRA and IC-LoRA plans with backend/Comfy route metadata.

## Install

Nexus BTA is developed and smoke-tested primarily on Windows with an NVIDIA/CUDA runtime. Linux and macOS notes are included so the repo can be evaluated or adapted on those systems, but not every generation or training route has the same hardware support.

### Windows

Windows is the recommended path for this repo today.

If this is a fresh machine, bootstrap the runtime first:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/bootstrap_nexus_runtime.ps1 -CopyPythonEnv
```

Then launch:

```bat
run.bat
```

Open:

```text
http://127.0.0.1:7861/ui
```

`run.bat`, `update.bat`, and the bootstrap script also check LTX Director dependencies. They install the WhatDreamsCost Director node, Kijai's Mel-Band RoFormer node, its Python requirements, the `MelBandRoformer_fp16.safetensors` model, and `ltx2.3-transition.safetensors` for LTX start/end transitions into the expected local folders when they are missing.

For the full LTX 2.3 video stack, use an NVIDIA GPU with CUDA. LTX's own open-source requirements list CUDA/NVIDIA-class hardware for local LTX 2.3 generation and training; the new Train LoRA UI can prepare LTX trainer jobs, but actual LTX LoRA/IC-LoRA training depends on the upstream LTX-2 trainer environment.

### Linux

Linux can run the backend and static UI manually. The bundled `.bat` and `.ps1` launchers are not portable yet, so create the runtime yourself:

```bash
python3.11 -m venv runtime/.venv
source runtime/.venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/comfyanonymous/ComfyUI.git runtime/ComfyUI
pip install -r runtime/ComfyUI/requirements.txt
python backend/run_backend.py
```

Open:

```text
http://127.0.0.1:7861/ui
```

Linux with an NVIDIA GPU and CUDA is the best non-Windows target for LTX 2.3 local generation, ComfyUI-LTXVideo routes and LTX-2 trainer jobs. Some Windows helper scripts for installing custom nodes still need to be translated to shell commands or handled through ComfyUI Manager.

### macOS

macOS support is partial. The legacy Nexus UI and backend can run, and ComfyUI officially supports Apple Silicon through PyTorch MPS, but the supported LTX 2.3 local video path is still CUDA/NVIDIA-focused. This is true even when you only want LTX 2.3 inference and do not need the LTX Trainer: LTX Desktop supports Apple Silicon Macs, but its current macOS generation path runs through the LTX API rather than local GPU inference. On Mac, expect SD/SDXL-style Comfy workflows to be the practical local target; use LTX API/LTX Desktop API mode or a remote CUDA machine for reliable LTX 2.3 video generation/training.

Apple Silicon setup:

```bash
python3.11 -m venv runtime/.venv
source runtime/.venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
git clone https://github.com/comfyanonymous/ComfyUI.git runtime/ComfyUI
pip install -r runtime/ComfyUI/requirements.txt
PYTORCH_ENABLE_MPS_FALLBACK=1 python backend/run_backend.py
```

Open:

```text
http://127.0.0.1:7861/ui
```

Mac compatibility notes:

- Apple Silicon/MPS can run many image workflows, but performance and custom-node coverage vary.
- Intel Mac is not a realistic target for local video generation.
- LTX 2.3 local generation without CUDA is not an officially supported Nexus target on macOS right now, even if you are not using the trainer. Experimental community MPS routes may appear, but expect broken custom nodes, black frames, missing kernels or CPU fallback until upstream LTX/Comfy dependencies support the exact route.
- LTX IC-LoRA workflows and LTX Trainer jobs should be treated as CUDA/NVIDIA-only locally.
- If `runtime/.venv/bin/python` is not picked up by your local settings, set `comfy_python` in `config/nexus_settings.json` to that path.

Reference notes checked while documenting macOS: ComfyUI Desktop for macOS supports Apple Silicon and recommends MPS; LTX's ComfyUI guide lists a CUDA-compatible GPU with 32GB+ VRAM for LTX-2 workflows; LTX Desktop's macOS generation currently runs through the LTX API instead of local GPU inference.

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
- LTX 2.3 latent upscale is part of the normal route. For a `512x512` output, Nexus samples the base latent at `256x256`, applies `LatentUpscaleModelLoader` + `LTXVLatentUpsampler` with `ltx-2.3-spatial-upscaler-x2-1.1.safetensors`, refines it, and then decodes the final video.
- LTX 2.3 start/end frame and transition-style txt2video/img2video use [joyfox/LTX-2.3-Transition-LORA](https://huggingface.co/joyfox/LTX-2.3-Transition-LORA) with trigger `zhuanchang`, installed under `models/loras/ltx_transition`.
- LTX 2.3 Motion Transfer uses IC-LoRA Union Control for Pose/Canny/Depth and the CameraMan IC-LoRA for camera motion transfer under `models/loras/ltx_ic`.
- LTX 2.3 Train LoRA uses the upstream [LTX-2 Trainer](https://github.com/Lightricks/LTX-2/tree/main/packages/ltx-trainer) for actual training. Nexus prepares local job configs for standard LTX LoRA, Motion LoRA, Audio-Video LoRA and IC-LoRA; running those jobs requires the upstream trainer environment and CUDA/NVIDIA hardware.
- Extras uses `models/upscale_models` for image/video raster upscalers, `models/video_restore_models` for FlashVSR/SeedVR2-style video restoration engines, `models/frame_interpolation` for RIFE/FILM frame interpolation, and `models/background_removal` for 2026 background-removal routes such as ComfyUI native BiRefNet. Alpha-aware video export is exposed as PNG sequence or MOV ProRes 4444.
- Remove BG expects `models/background_removal/birefnet.safetensors` for the native ComfyUI BiRefNet workflow. RMBG-2.0, InSPyReNet and BEN/BEN2 remain selectable compatibility routes when matching ComfyUI custom nodes are installed.

## Credits

- LTX 2.3 Director integration credits: [WhatDreamsCost/WhatDreamsCost-ComfyUI](https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI).
- LTX 2.3 Transition LoRA credits: [joyfox/LTX-2.3-Transition-LORA](https://huggingface.co/joyfox/LTX-2.3-Transition-LORA).

## Output Layout

Generated media is grouped by type:

```text
output/image/YYYYMMDD_HHMMSS_<preset>_<activity>_...
output/video/YYYYMMDD_HHMMSS_<preset>_<activity>_...
output/videos/YYYYMMDD_HHMMSS_LTX_DIRECTOR_SEGMENTS_...   # Joined Director renders
output/director/YYYYMMDD_HHMMSS/segments/...              # Per-segment Director videos
output/extras/video/YYYYMMDD_HHMMSS/...   # Extras PNG sequences
```

Nexus applies this naming layer to default templates, loaded workflows and visual workflow overrides before sending the job to ComfyUI. Normal video generations stay in the `output/video` root. LTX Director segment renders are preserved in the Director archive and the joined timeline video is exported to `output/videos`. Extras video PNG sequence exports receive a per-run dated folder under `output/extras/video` so frame sequences do not mix with other runs. The Gallery can navigate output folders with visible folder cards, back/forward history, optional folder picker and date/type sorting.

## Verified Smoke Battery

Last verified on May 29, 2026:

- SD 1.5 and SDXL image generation, img2img, inpaint, and ControlNet Canny.
- Qwen, Flux and Z-Image/ZImage ControlNet routes at 512x512 with side-menu model selection.
- WAN 2.2 at 512x512, 2 seconds, 24 FPS.
- LTX 2.3 Linear View at 512x512/768x512, 24 FPS, 4 steps, CFG 1, with latent upscale x2, start/end frame, Transition LoRA, IC Detailer and Motion Transfer modes for Pose, Canny, Depth and Camera/Cameraman.
- LTX 2.3 Director Suite with Motion Transfer segments, non-motion segments, Transition LoRA end frames, CameraMan motion transfer, IC identity conditioning, per-segment outputs and joined final video export.
- Extras image upscale with Remacri/UltraSharp/RealESRGAN, RIFE `rife_v4.26`, video upscale, LTX IC Detailer refine/upscale, FlashVSR/SeedVR2-ready engines, denoise, face restoration, PNG sequence alpha and MOV ProRes 4444 alpha export.
- Extras Remove BG frontend plan sync with `models/background_removal/birefnet.safetensors`, recommended `Remove BG Image` and `Remove BG Video` presets, video PNG sequence or MOV ProRes 4444 alpha export, RGBA/mask output options and folder-aware Gallery navigation.
- Inpaint with LanPaint default workflow, Differential Diffusion option, paint/remove masks, generative outpaint canvas expansion, magic wand/select object and undo/redo.
- LTX 2.3 Director Suite with custom background audio, generated speech/ambience segments, per-segment negative prompts and non-black video output.
- Legacy Train LoRA front end across SD, SDXL, Illustrious, Pony, Flux, Flux 2, Flux 2 Klein, Qwen, WAN, LTX 2.3, Anima, Z-Image Turbo and Lumina templates, including LTX character/style/motion/audio-video/IC-LoRA job preparation with Launch disabled.
- Anima with Concept LoRA selection and gallery metadata.
- Node Workflow editor smoke: menu search, categorized add-node menu, visual multi-selection, grouped drag behavior, port connection/unlink affordances, and workflow override routing from the active graph.

## Notes

- Generated media stays in `output/image` and `output/video`.
- Temporary inputs, masks, Comfy previews, and Nexus temp files are cleaned after generation.
- Runtime files, model weights, generated media, and local settings are ignored by git.

## License

MIT. See [LICENSE](LICENSE).
