# NexusBTA v0.2.46

`NexusBTA v0.2.46 - LTX 2.5, MiniMax H3 latent bridge and Image Edit`

Short release summary:

- MiniMax H3 reference videos are normalized to the FPS and resolution selected in the UI.
- Frame loading stops at the H3-aligned frame count before VAE and Qwen processing.
- Reference audio is limited to the same normalized time window.
- Video Helper Suite is installed only when missing and uses the configured custom-nodes path.
- Train LoRA now launches the correct Anima Kohya script, writes a valid training TOML, and preserves UTF-8 logs on Windows.
- Krea 2 adds an isolated local text-to-image and three-image style-reference route using ComfyUI's native Krea 2 nodes.
- Krea 2 model downloads are opt-in and respect the configured models path.
- Krea 2 detects the available RTX profile and writes a native Comfy model-path map for custom model folders on the next runtime start.
- LTX 2.5 adds an isolated native audio/video workflow with MultiShot, spatial latent upscale and optional temporal latent upscale.
- MiniMax H3 latent upscale uses the local LTX 2.5 bridge only when enabled, preserving the native H3 workflow when disabled.
- MiniMax H3 REF2VA adds a synchronized one-image Image Edit route with editable sampling steps (8 by default) and first-frame PNG output.
- Silent MiniMax H3 reference videos receive a temporary AAC track so VHS audio extraction cannot fail after sampling completes.

Complete version:

[docs/releases/v0.2.46.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.46/docs/releases/v0.2.46.md)
