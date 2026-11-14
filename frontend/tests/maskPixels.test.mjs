import assert from 'node:assert/strict';
import test from 'node:test';
import { fittedMaskSize, historyLimit, binaryMaskPixels, clampViewportPan } from '../src/views/home/maskPixels.ts';

test('mask size retains aspect ratio and does not upscale smaller sources', () => {
  assert.deepEqual(fittedMaskSize(4096, 2160), { width: 1536, height: 810 });
  assert.deepEqual(fittedMaskSize(640, 480), { width: 640, height: 480 });
  assert.deepEqual(fittedMaskSize(NaN, 0), { width: 1536, height: 1536 });
});

test('history is bounded by both actions and approximate RGBA byte budget', () => {
  assert.equal(historyLimit(256, 256), 12);
  assert.equal(historyLimit(1536, 1536), 3);
  assert.equal(historyLimit(4096, 4096), 1);
});

test('binary mask preserves black and white PNG representation without leaks', () => {
  const data = new Uint8ClampedArray([55, 66, 77, 0, 16, 14, 11, 3, 99, 12, 13, 4, 1, 2, 3, 255]);
  assert.equal(binaryMaskPixels(data), true);
  assert.deepEqual([...data], [0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255, 255, 255, 255, 255, 255]);
  assert.throws(() => binaryMaskPixels(new Uint8ClampedArray([1, 2, 3])), /RGBA/);
  assert.equal(binaryMaskPixels(new Uint8ClampedArray([0, 0, 0, 0])), false);
});

test('zoom/pan is bounded and reset view cannot drift from mask alignment', () => {
  assert.deepEqual(clampViewportPan(1000, -1000, 2, 400, 240), { x: 200, y: -120 });
  assert.deepEqual(clampViewportPan(30, 30, 1, 400, 240), { x: 0, y: 0 });
  assert.deepEqual(clampViewportPan(Number.NaN, Number.POSITIVE_INFINITY, 3, 120, 80), { x: 0, y: 0 });
});
