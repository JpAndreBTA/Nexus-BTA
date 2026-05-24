# Nexus BTA Version Tag Notes

Suggested tag: `v0.1.1-output-extras`

Release text:

Nexus BTA now organizes generated media into `output/image` and `output/video` with date/time filename prefixes across default templates, loaded Comfy workflows and visual Node Workflow overrides. Normal video generations stay in the `output/video` root, while Extras PNG sequence exports receive a dated folder under `output/extras/video`. Extras gained alpha-aware image/video export options, RIFE `rife_v4.26` interpolation support, video upscale checks, image upscale checks, 2026 Remove BG Image/Video presets backed by `models/background_removal/birefnet.safetensors`, and folder-aware Gallery navigation. The launcher remains English-only and the smoke battery covers frontend sync, Comfy model discovery, RIFE alpha PNG sequence, MOV ProRes 4444 alpha video upscale, Remove BG frontend routing and inpaint workflow synchronization.
