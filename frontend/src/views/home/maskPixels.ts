// Pure pixel and budget functions shared with the native/WASM mask pipeline.
// Keep the editor source-size cap, mask alpha threshold and undo budget tested.
export const MAX_EDIT_EDGE = 1536;
export const MAX_HISTORY_STEPS = 12;
export const MAX_HISTORY_BYTES = 32 * 1024 * 1024;

export function fittedMaskSize(width: number, height: number): { width: number; height: number } {
  const sourceWidth = Number.isFinite(width) && width > 0 ? width : MAX_EDIT_EDGE;
  const sourceHeight = Number.isFinite(height) && height > 0 ? height : MAX_EDIT_EDGE;
  const scale = Math.min(1, MAX_EDIT_EDGE / Math.max(sourceWidth, sourceHeight));
  return {
    width: Math.max(16, Math.round(sourceWidth * scale)),
    height: Math.max(16, Math.round(sourceHeight * scale)),
  };
}

export function historyLimit(width: number, height: number): number {
  const bytes = Math.max(1, width * height * 4);
  return Math.max(1, Math.min(MAX_HISTORY_STEPS, Math.floor(MAX_HISTORY_BYTES / bytes)));
}

/** Overwrite a scratch RGBA buffer into legacy opaque black/white mask format. */
export function binaryMaskPixels(pixels: Uint8ClampedArray): boolean {
  if (pixels.length % 4 !== 0) throw new Error('Expected RGBA pixel buffer.');
  let hasMask = false;
  for (let index = 0; index < pixels.length; index += 4) {
    const marked = pixels[index + 3] > 3;
    const value = marked ? 255 : 0;
    pixels[index] = value;
    pixels[index + 1] = value;
    pixels[index + 2] = value;
    pixels[index + 3] = 255;
    hasMask ||= marked;
  }
  return hasMask;
}

/** CSS pan limits, in viewport pixels, keep the zoomed image over the editor. */
export function clampViewportPan(
  x: number, y: number, zoom: number, viewportWidth: number, viewportHeight: number,
): { x: number; y: number } {
  if (!Number.isFinite(zoom) || zoom <= 1) return { x: 0, y: 0 };
  const allowedX = Math.max(0, ((zoom - 1) * Math.max(0, viewportWidth)) / 2);
  const allowedY = Math.max(0, ((zoom - 1) * Math.max(0, viewportHeight)) / 2);
  return {
    x: Math.max(-allowedX, Math.min(allowedX, Number.isFinite(x) ? x : 0)),
    y: Math.max(-allowedY, Math.min(allowedY, Number.isFinite(y) ? y : 0)),
  };
}
