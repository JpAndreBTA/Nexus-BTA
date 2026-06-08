# NexusBTA v0.2.27

`NexusBTA v0.2.27 - Offline UI Hotfix, xFormers Runtime and Ideogram-4 Stability`

Short release summary:

- Offline UI: `/ui` starts with local Tailwind, fonts, Font Awesome, Three.js, MediaPipe WASM and pose model assets.
- Backend sync: offline browser mode still reaches local Nexus and embedded ComfyUI APIs.
- Dependency guard: missing online assets now ask users to reconnect or install manually instead of failing silently.
- Runtime: xFormers installs through normal pip resolution and skips network when the runtime is already import-ready.
- PowerShell: native pip retry warnings are handled without red `NativeCommandError` noise.
- Ideogram-4: functional frontend route, asset status/download contract and regional prompt controls are included.

Complete version:

[docs/releases/v0.2.27.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.27/docs/releases/v0.2.27.md)
