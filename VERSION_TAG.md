# NexusBTA v0.2.29

`NexusBTA v0.2.29 - Blackwell xFormers Runtime Hotfix`

Short release summary:

- Runtime attention: xFormers availability now requires a real CUDA `memory_efficient_attention` kernel probe, not only import success.
- RTX 5090 / Blackwell: broken xFormers wheels are detected before generation so auto/Sage paths can fall back instead of crashing with `memory_efficient_attention_forward`.
- Runtime capabilities: `/api/runtime/capabilities` now reports the actual xFormers operator error for diagnostics.
- Hotfix scope: keeps the Ideogram-4 functional behavior from `v0.2.28`.

Complete version:

[docs/releases/v0.2.29.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.29/docs/releases/v0.2.29.md)
