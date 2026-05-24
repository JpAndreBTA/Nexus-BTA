# Nexus BTA Model Assets

This file is a download guide for model weights, LoRAs, distilled adapters and textual inversion files. Do not commit downloaded weights to git; keep them under `models/`.

## Folder Map

```text
models/checkpoints        SD 1.5, SDXL, Pony, Illustrious, Anima and full checkpoint files
models/unet               Optional UNET/diffusion-only models; Nexus also detects these from checkpoints
models/diffusion_models   Optional diffusion-model layout used by some ComfyUI workflows
models/loras              LoRA, LyCORIS and distilled LoRA files
models/embeddings         Textual inversion / embedding files
models/vae                SD, SDXL, Flux, Wan and LTX VAE files
models/text_encoders      CLIP, T5, UMT5, Gemma and Qwen text encoders
models/clip_vision        Required CLIP vision encoders for WAN first/end-frame conditioning
models/controlnet         SD 1.5 and SDXL ControlNet models
models/upscale_models     ESRGAN/RealESRGAN-style upscalers
models/latent_upscale_models  LTX latent upscalers
models/diffusion_models/MelRoFormer  Mel-Band RoFormer audio cleanup model for LTX Director audio workflows
```

## Core Model Links

| Family | Recommended source | Place under |
| --- | --- | --- |
| SD 1.5 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) or a compatible checkpoint | `models/checkpoints/sd15` |
| SDXL / Pony / Illustrious | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) or compatible finetunes | `models/checkpoints/sdxl` |
| Flux / Flux.1 | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) plus Flux AE/text encoders | `models/checkpoints/flux`, `models/vae`, `models/text_encoders` |
| Flux.2 Dev | [black-forest-labs/FLUX.2-dev](https://github.com/black-forest-labs/flux2) or Comfy split files | `models/checkpoints/flux` or `models/diffusion_models`, `models/text_encoders/mistral_3_small_flux2_bf16.safetensors`, `models/vae/flux2-vae.safetensors` |
| Flux.2 Klein | [FLUX.2 Klein 4B/9B](https://docs.comfy.org/tutorials/flux/flux-2-klein) | `models/checkpoints/flux` or `models/diffusion_models`, 4B: `qwen_3_4b.safetensors`, 9B: `qwen_3_8b_fp8mixed.safetensors`, plus `models/vae/flux2-vae.safetensors` |
| WAN 2.2 | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B), [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B), [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B), compatible CLIP Vision such as `clip_vision_h.safetensors` | `models/checkpoints/wan`, `models/text_encoders`, `models/vae`, `models/clip_vision` |
| LTX 2.3 | [Lightricks/LTX-2.3](https://huggingface.co/Lightricks/LTX-2.3) | `models/checkpoints/ltx`, `models/loras/ltx`, `models/latent_upscale_models`, `models/vae`, `models/text_encoders` |

## Lightweight LoRA Starter List

| Family | Asset | Notes |
| --- | --- | --- |
| SD 1.5 | [latent-consistency/lcm-lora-sdv1-5](https://huggingface.co/latent-consistency/lcm-lora-sdv1-5) | Lightweight acceleration LoRA. Use low steps and low CFG. |
| SD 1.5 / SDXL | [ByteDance/Hyper-SD](https://huggingface.co/ByteDance/Hyper-SD) | Pick the LoRA file that matches the base family and intended step count. |
| SDXL / Pony / Illustrious | [latent-consistency/lcm-lora-sdxl](https://huggingface.co/latent-consistency/lcm-lora-sdxl) | SDXL LCM LoRA; good for fast smoke tests. |
| SDXL / Pony / Illustrious | [ByteDance/SDXL-Lightning](https://huggingface.co/ByteDance/SDXL-Lightning) | Download the LoRA variant that matches the exact step count. |
| Illustrious | [Stableyogi/Detail-Tweaker-Illustrious](https://huggingface.co/Stableyogi/Detail-Tweaker-Illustrious) | Detail/quality LoRA for Illustrious-compatible routes. |
| Flux | [alimama-creative/FLUX.1-Turbo-Alpha](https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha) | 8-step distilled Flux LoRA; recommended guidance is low. |
| Flux | [XLabs-AI/flux-lora-collection](https://huggingface.co/XLabs-AI/flux-lora-collection/tree/main) | Small style LoRAs such as anime, disney, mjv6, realism and scenery. |
| Flux anime | [aionthegrind/anime-lora](https://huggingface.co/aionthegrind/anime-lora) | Flux anime style LoRA; trigger word is listed on the model card. |
| Anima | [RedRayz/my-anima-lora](https://huggingface.co/RedRayz/my-anima-lora) | Native Anima LoRA examples. Keep these in an Anima subfolder. |

Suggested folders:

```text
models/loras/sd15
models/loras/sdxl
models/loras/flux
models/loras/anima
models/loras/wan
models/loras/ltx
```

## Textual Inversion / Embeddings

Put `.safetensors`, `.pt` or `.bin` embeddings in:

```text
models/embeddings
```

Starter negative embeddings:

| Family | Asset | Token / note |
| --- | --- | --- |
| SD 1.5 | [EvilEngine/easynegative](https://huggingface.co/EvilEngine/easynegative) | `embedding:easynegative` in Nexus, normally for the negative prompt. |
| SD 1.5 / SDXL utility pack | [Drditone/Textual-Inversion](https://huggingface.co/Drditone/Textual-Inversion/tree/main) | Includes several SafeTensors negative embeddings; match SD 1.5 vs XL carefully. |

Nexus enables the embedding picker for SD 1.5, SDXL, Pony and Illustrious-compatible image routes. WAN 2.2 and LTX 2.3 use video-specific encoders, so classic textual inversion is intentionally disabled there; use LoRA or distilled LoRA slots instead.

## Distilled / Video Notes

- WAN 2.2 smoke preset: `512x512`, `5s`, `16 FPS`, 4 steps, CFG 1.0 for the high/low route.
- WAN 2.2 14B I2V/T2V uses `wan_2.1_vae.safetensors`; reserve `wan22-vae` / `wan2.2_vae` for TI2V 5B-style routes.
- WAN 2.2 4-step quality depends on the matching high-noise and low-noise distilled/LightX2V LoRA pair in `models/loras/wan`. Nexus auto-detects files whose names include `high`/`low` plus `lightx2v`, `4step`, `4-step`, `lightning` or `distill`.
- WAN 2.2 first/last-frame mode uses `WanFirstLastFrameToVideo` as motion conditioning, not a post-video crossfade. Keep a compatible CLIP Vision model in `models/clip_vision`; Nexus wires `CLIPVisionEncode` for the start and end images so the model sees both visual anchors.
- The Flux template auto-detects Flux.1 versus Flux.2/Flux.2 Klein from the selected model name. Flux.1 keeps the CLIP-L + T5 route; Flux.2 uses Comfy's `CLIPLoader(type=flux2)`, `Flux2Scheduler`, `EmptyFlux2LatentImage`, Flux.2 VAE, and model-only Flux.2 LoRAs.
- LTX 2.3 smoke preset: `512x512`, `4s`, `24 FPS`; distilled checkpoints and distilled LoRA variants usually run at low CFG and short step counts.
- LTX 2.3 latent upscale is expected for normal `512x512` outputs: Nexus samples the base video latent at half resolution, runs the spatial latent upscaler/refiner, then decodes the final frames. A `256x256` result usually means the smoke test explicitly selected `None` or a `256x256` output size.
- LTX 2.3 assets include full, distilled, distilled LoRA, spatial upscaler and temporal upscaler variants on the official Hugging Face repository.
- LTX 2.3 Director audio workflows that follow WhatDreamsCost v30 use Kijai's `ComfyUI-MelBandRoFormer` custom node and `MelRoFormer/MelBandRoformer_fp16.safetensors` under `models/diffusion_models`. This keeps source/background audio and generated speech/ambience from being routed as raw noisy audio latents.
- `run.bat`, `update.bat`, and `scripts/bootstrap_nexus_runtime.ps1` run `scripts/install_ltx_director_deps.ps1`, which installs the LTX Director custom-node dependencies and downloads `MelBandRoformer_fp16.safetensors` to `models/diffusion_models/MelRoFormer` if it is missing.
- Keep WAN high-noise and low-noise model files together so Nexus can pick the paired route automatically. A style LoRA without a 4-step/distill token is treated as a user Concept LoRA, not as the required fast adapter.

## Download Helper

Install the Hugging Face CLI when you want direct downloads:

```powershell
pip install "huggingface_hub[cli]"
huggingface-cli download latent-consistency/lcm-lora-sdv1-5 --local-dir models/loras/sd15/lcm-lora-sdv1-5
```

For large gated repositories, sign in first:

```powershell
huggingface-cli login
```
