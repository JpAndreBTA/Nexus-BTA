# Nexus Image Core (experimental)

Deterministic Rust operations intended for future accelerated mask preparation and CPU-bound media transformations. **Not integrated into the production browser path yet.** The existing JS canvas remains the fallback.

Native validation: `cargo test --manifest-path crates/nexus-image-core/Cargo.toml`.

Optional browser build (requires Rust target and wasm-pack): `wasm-pack build crates/nexus-image-core --target web --features wasm`.
Inspect and benchmark the generated bundle before integrating it into Rsbuild. Do not commit `pkg/` or `target/` and do not move ComfyUI/PyTorch inference into WASM.
