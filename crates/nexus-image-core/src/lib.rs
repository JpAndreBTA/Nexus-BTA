//! Deterministic image/mask operations shared by native and future WebAssembly clients.
//! This crate is experimental and NOT wired to the existing browser inpainting path.

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

/// Convert an RGBA overlay (alpha marks masked areas) into Comfy-compatible
/// opaque black/white RGBA. Returns empty for malformed RGBA buffers.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn alpha_overlay_to_binary_mask(rgba: &[u8], alpha_threshold: u8) -> Vec<u8> {
    if rgba.len() % 4 != 0 {
        return Vec::new();
    }
    let mut output = Vec::with_capacity(rgba.len());
    for pixel in rgba.chunks_exact(4) {
        let mark = if pixel[3] > alpha_threshold { 255 } else { 0 };
        output.extend_from_slice(&[mark, mark, mark, 255]);
    }
    output
}

/// Detect whether at least one pixel is marked, avoiding an unnecessary export.
#[cfg_attr(feature = "wasm", wasm_bindgen)]
pub fn has_masked_pixels(rgba: &[u8], alpha_threshold: u8) -> bool {
    rgba.chunks_exact(4).any(|pixel| pixel[3] > alpha_threshold)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn marks_alpha_and_ignores_rgb() {
        let result = alpha_overlay_to_binary_mask(&[100, 200, 30, 0, 3, 9, 99, 180], 3);
        assert_eq!(result, vec![0, 0, 0, 255, 255, 255, 255, 255]);
    }
    #[test]
    fn rejects_incomplete_pixels() {
        assert!(alpha_overlay_to_binary_mask(&[1, 2, 3], 3).is_empty());
    }
    #[test]
    fn checks_threshold() {
        assert!(!has_masked_pixels(&[255, 255, 255, 3], 3));
        assert!(has_masked_pixels(&[255, 255, 255, 4], 3));
    }
}
