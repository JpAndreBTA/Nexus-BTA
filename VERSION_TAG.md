# NexusBTA v0.2.43

`NexusBTA v0.2.43 - MiniMax H3 and Train LoRA maintenance`

Short release summary:

- MiniMax H3 reference videos are normalized to the FPS and resolution selected in the UI.
- Frame loading stops at the H3-aligned frame count before VAE and Qwen processing.
- Reference audio is limited to the same normalized time window.
- Video Helper Suite is installed only when missing and uses the configured custom-nodes path.
- Train LoRA now launches the correct Anima Kohya script, writes a valid training TOML, and preserves UTF-8 logs on Windows.

Complete version:

[docs/releases/v0.2.43.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.43/docs/releases/v0.2.43.md)
