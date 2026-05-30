# NexusBTA v0.2.20

Suggested tag: `v0.2.20`

Release title:

`NexusBTA v0.2.20 - Customize, Workflow Nodes and 3D Hotfix`

Release text:

NexusBTA v0.2.20 includes the full v0.2.19 release package plus a UI/template hotfix for the bugs found after release.

## Included From v0.2.19

- Added Model 3D / TRELLIS workspace updates.
- Added texture paint and UV workflow improvements.
- Added template save/load/customization tools.
- Added workflow node pin, bypass, color and reorder fixes.
- Added preset-aware model, VAE, CLIP/Text Encoder and LoRA selectors.
- Added backend synchronization for edited workflow nodes and pinned nodes.
- Added SageAttention and xFormers runtime acceleration handling.
- Added low-VRAM TRELLIS/FlexGEMM hotfix support.
- Improved startup, model path sync and runtime optimization.

## New In v0.2.20

- Fixed side-menu panels showing hidden technical controls by mistake.
- Fixed broken duplicated layout info in Image Guidance, video and LoRA panels.
- Fixed Nexus Customize colors not reaching prompts, buttons, toggles, tabs and modals.
- Fixed positive and negative prompt text/placeholder colors.
- Fixed Add LoRA and Civitai modals keeping black surfaces outside the selected theme.
- Added a modern theme randomizer button inside NexusBTA Customize.
- Improved template JSON compatibility with a new `customization` section.
- Kept old template JSON files working through automatic migration.
- Updated README with clearer beginner-friendly instructions.
- Added Ko-fi support button: https://ko-fi.com/jpandre
- Added README examples for Nexus Customize and the 3D workspace.
- Changed README examples to show two images per row on GitHub.

## 3D Status

- The 3D workspace is still experimental.
- It is useful for testing image-to-3D, TRELLIS, texture paint and UV routes.
- It is not fully optimized yet.
- Some 3D bugs are known and will be fixed in the next updates.

## Recommended Update

- Use this release instead of v0.2.19 if you want the Customize UI, modals, prompts and template JSON fixes.
- Existing template JSON files should continue to load.
- New template JSON files now save theme, panel order, panel collapse state, pinned nodes and customization data in a cleaner format.
