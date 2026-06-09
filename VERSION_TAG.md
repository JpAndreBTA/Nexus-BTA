# NexusBTA v0.2.30

`NexusBTA v0.2.30 - Runtime Attention Settings Hotfix`

Short release summary:

- Runtime settings: SageAttention, xFormers and PyTorch SDPA toggles now save as independent preferences.
- Compatibility sets: users can keep Sage + xFormers, Sage + SDPA, xFormers + SDPA or SDPA-only from the Settings UI.
- Backend runtime: SDPA preference is persisted with `enable_pytorch_attention` and included in runtime restart signatures.
- Hotfix scope: keeps the Blackwell xFormers operator probe from `v0.2.29`.

Complete version:

[docs/releases/v0.2.30.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.30/docs/releases/v0.2.30.md)
