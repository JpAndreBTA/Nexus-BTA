# NexusBTA v0.2.28

`NexusBTA v0.2.28 - Ideogram-4 Functional Hotfix`

Short release summary:

- Ideogram-4: regional boxes now use the native KJ `elements_data` contract instead of leaking guide JSON into generation.
- Ideogram-4 txt2img/img2img: text regions, object regions and reference-preserve prompts stay synchronized with the backend.
- Ideogram-4 output guard: unintended marks, borders, watermarks and random text are blocked unless the prompt or region explicitly asks for brand/logo/signage details.
- Legacy `/ui`: Ideogram region overlays remain editor-only and do not cover generated outputs.
- Hotfix scope: keeps the offline UI, xFormers runtime and dependency behavior from `v0.2.27`.

Complete version:

[docs/releases/v0.2.28.md](https://github.com/JpAndreBTA/Nexus-BTA/blob/v0.2.28/docs/releases/v0.2.28.md)
