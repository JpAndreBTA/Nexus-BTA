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
models/controlnet         SD 1.5 and SDXL ControlNet models
models/upscale_models     ESRGAN/RealESRGAN-style upscalers
models/latent_upscale_models  LTX latent upscalers
```

## Core Model Links

| Family | Recommended source | Place under |
| --- | --- | --- |
| SD 1.5 | [stable-diffusion-v1-5/stable-diffusion-v1-5](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5) or a compatible checkpoint | `models/checkpoints/sd15` |
| SDXL / Pony / Illustrious | [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) or compatible finetunes | `models/checkpoints/sdxl` |
| Flux | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) plus Flux VAE/text encoders | `models/checkpoints/flux`, `models/vae`, `models/text_encoders` |
| WAN 2.2 | [Wan-AI/Wan2.2-I2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B), [Wan-AI/Wan2.2-T2V-A14B](https://huggingface.co/Wan-AI/Wan2.2-T2V-A14B), [Wan-AI/Wan2.2-TI2V-5B](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B) | `models/checkpoints/wan`, `models/text_encoders`, `models/vae` |
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

- WAN 2.2 smoke preset: `512x512`, `2s`, `24 FPS`, low CFG, low step count.
- LTX 2.3 smoke preset: `512x512`, `4s`, `24 FPS`; distilled checkpoints and distilled LoRA variants usually run at low CFG and short step counts.
- LTX 2.3 assets include full, distilled, distilled LoRA, spatial upscaler and temporal upscaler variants on the official Hugging Face repository.
- Keep WAN high-noise and low-noise model files together so Nexus can pick the paired route automatically.

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
