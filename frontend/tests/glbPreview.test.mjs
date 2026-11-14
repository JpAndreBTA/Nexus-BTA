import assert from 'node:assert/strict';
import test from 'node:test';
import { MAX_PREVIEW_GLB_BYTES, parseGlbPreview } from '../src/views/gallery/glbPreview.ts';

function pad4(input) {
  const raw = Buffer.from(input);
  return Buffer.concat([raw, Buffer.alloc((4 - (raw.length % 4)) % 4, 0x20)]);
}
function makeGlb() {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const json = {
    asset: { version: '2.0' }, buffers: [{ byteLength: positions.byteLength }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
    accessors: [{ bufferView: 0, byteOffset: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] }],
  };
  const js = pad4(JSON.stringify(json));
  const bin = Buffer.from(positions.buffer);
  const out = Buffer.alloc(12 + 8 + js.length + 8 + bin.length);
  out.writeUInt32LE(0x46546c67, 0);
  out.writeUInt32LE(2, 4);
  out.writeUInt32LE(out.length, 8);
  out.writeUInt32LE(js.length, 12);
  out.writeUInt32LE(0x4e4f534a, 16);
  js.copy(out, 20);
  const start = 20 + js.length;
  out.writeUInt32LE(bin.length, start);
  out.writeUInt32LE(0x004e4942, start + 4);
  bin.copy(out, start + 8);
  return out;
}
function toArrayBuffer(buffer) {
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength);
}
test('GLB parser builds a bounded triangle with finite normals and camera extent', () => {
  const mesh = parseGlbPreview(toArrayBuffer(makeGlb()));
  assert.equal(mesh.triangles, 1);
  assert.equal(mesh.positions.length, 9);
  assert.equal(mesh.normals.length, 9);
  assert.deepEqual(mesh.center, [0.5, 0.5, 0]);
  assert.equal(mesh.extent, 1);
  assert.ok(mesh.normals.every(Number.isFinite));
});
test('GLB parser rejects invalid header, external references and truncated chunks', () => {
  const valid = makeGlb();
  const corrupt = Buffer.from(valid); corrupt.writeUInt32LE(1, 4);
  assert.throws(() => parseGlbPreview(toArrayBuffer(corrupt)), /glTF 2.0/);
  const corruptSize = Buffer.from(valid); corruptSize.writeUInt32LE(1, 12);
  assert.throws(() => parseGlbPreview(toArrayBuffer(corruptSize)), /chunk/);
  assert.throws(() => parseGlbPreview(toArrayBuffer(valid.subarray(0, 32))));
  assert.throws(() => parseGlbPreview(new ArrayBuffer(MAX_PREVIEW_GLB_BYTES + 1)), /24 MiB/);
});
