# NexusBTA v0.2.40

`NexusBTA v0.2.40 - Gallery workflow isolation hotfix`

Short release summary:

- Gallery handoff: txt2img receives only the positive and negative prompts.
- Reference handoff: img2img, inpaint and img2video receive only prompts and the selected image or video.
- Workflow isolation: VAE, text encoder, video/audio VAE, Concept LoRAs, Lightning LoRAs and distilled LoRAs remain owned by the active template.
- Template safety: sending an older gallery output no longer contaminates another model workflow.

Complete version:

[docs/releases/v0.2.40.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.40/docs/releases/v0.2.40.md)
