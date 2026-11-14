/**
 * Strict, dependency-free GLB v2 preview parser.
 * Preview ONE triangle primitive only; no materials, skinning, external buffers
 * or custom glTF extensions. Never execute imported scripts.
 */
export interface PreviewGeometry {
  positions: Float32Array;
  normals: Float32Array;
  center: [number, number, number];
  extent: number;
  triangles: number;
}

export const MAX_PREVIEW_GLB_BYTES = 24 * 1024 * 1024;
const MAX_TRIANGLES = 65_536;

interface GltfBufferView { buffer?: number; byteOffset?: number; byteLength: number; byteStride?: number }
interface GltfAccessor {
  bufferView?: number; byteOffset?: number; componentType: number;
  count: number; type: string; sparse?: unknown;
}
interface GltfPrimitive { attributes?: Record<string, number>; indices?: number; mode?: number }
interface GltfDocument {
  meshes?: Array<{ primitives?: GltfPrimitive[] }>;
  accessors?: GltfAccessor[];
  bufferViews?: GltfBufferView[];
  buffers?: Array<{ byteLength: number; uri?: string }>;
}

function integerInRange(value: unknown, low: number, high: number): value is number {
  return typeof value === 'number' && Number.isSafeInteger(value) && value >= low && value <= high;
}

function readAccessor(
  document: GltfDocument, binary: Uint8Array,
  accessorIndex: number, position: boolean,
): { count: number; get: (index: number, component: number) => number } {
  const accessor = document.accessors?.[accessorIndex];
  const view = document.bufferViews?.[accessor?.bufferView ?? -1];
  if (!accessor || !view || accessor.sparse !== undefined || view.buffer !== 0) {
    throw new Error('Preview supports embedded, non-sparse GLB accessors only.');
  }
  const count = accessor.count;
  const componentSize = accessor.componentType === 5126 || accessor.componentType === 5125 ? 4
    : accessor.componentType === 5123 ? 2 : accessor.componentType === 5121 ? 1 : 0;
  const components = position ? 3 : 1;
  if (
    !componentSize ||
    !integerInRange(count, 1, position ? MAX_TRIANGLES * 3 : MAX_TRIANGLES * 3) ||
    (position && (accessor.type !== 'VEC3' || accessor.componentType !== 5126)) ||
    (!position && (accessor.type !== 'SCALAR' || ![5121, 5123, 5125].includes(accessor.componentType)))
  ) {
    throw new Error('Unsupported GLB geometry accessor.');
  }
  const viewOffset = view.byteOffset ?? 0;
  const accessorOffset = accessor.byteOffset ?? 0;
  const stride = view.byteStride ?? componentSize * components;
  if (
    !integerInRange(viewOffset, 0, binary.byteLength) ||
    !integerInRange(view.byteLength, 1, binary.byteLength - viewOffset) ||
    !integerInRange(accessorOffset, 0, view.byteLength) ||
    !integerInRange(stride, componentSize * components, 256) ||
    (accessor.componentType === 5126 && (viewOffset + accessorOffset) % 4 !== 0) ||
    accessorOffset + (count - 1) * stride + componentSize * components > view.byteLength
  ) {
    throw new Error('GLB accessor exceeds bounded binary view.');
  }
  const data = new DataView(binary.buffer, binary.byteOffset, binary.byteLength);
  const start = viewOffset + accessorOffset;
  return {
    count,
    get: (i, c) => {
      const offset = start + i * stride + c * componentSize;
      if (accessor.componentType === 5126) return data.getFloat32(offset, true);
      if (accessor.componentType === 5125) return data.getUint32(offset, true);
      if (accessor.componentType === 5123) return data.getUint16(offset, true);
      return data.getUint8(offset);
    },
  };
}

export function parseGlbPreview(buffer: ArrayBuffer): PreviewGeometry {
  if (buffer.byteLength < 28 || buffer.byteLength > MAX_PREVIEW_GLB_BYTES) {
    throw new Error('GLB preview requires a local file of at most 24 MiB.');
  }
  const view = new DataView(buffer);
  if (view.getUint32(0, true) !== 0x46546c67 || view.getUint32(4, true) !== 2 || view.getUint32(8, true) !== buffer.byteLength) {
    throw new Error('Only valid glTF 2.0 GLB files are supported.');
  }
  let offset = 12;
  let json: GltfDocument | undefined;
  let binary: Uint8Array | undefined;
  while (offset + 8 <= buffer.byteLength) {
    const chunkSize = view.getUint32(offset, true);
    const kind = view.getUint32(offset + 4, true);
    offset += 8;
    if (chunkSize > buffer.byteLength - offset || chunkSize % 4 !== 0) {
      throw new Error('Malformed GLB chunk bounds.');
    }
    if (kind === 0x4e4f534a && json === undefined) {
      json = JSON.parse(new TextDecoder('utf-8', { fatal: true })
        .decode(new Uint8Array(buffer, offset, chunkSize)).replace(/\u0000+$/, '')) as GltfDocument;
    }
    if (kind === 0x004e4942 && binary === undefined) binary = new Uint8Array(buffer, offset, chunkSize);
    offset += chunkSize;
  }
  if (offset !== buffer.byteLength || !json || !binary || !json.meshes?.length || !json.buffers?.length ||
      json.buffers[0].uri || !integerInRange(json.buffers[0].byteLength, 1, binary.byteLength)) {
    throw new Error('GLB is incomplete or references unsupported external buffers.');
  }
  // A deliberately scoped preview of the FIRST triangle primitive.
  const primitive = json.meshes.flatMap((mesh) => mesh.primitives ?? [])
    .find((mesh) => (mesh.mode ?? 4) === 4 && mesh.attributes?.POSITION !== undefined);
  if (!primitive || !integerInRange(primitive.attributes?.POSITION, 0, 1_000_000)) {
    throw new Error('No supported GLB triangle mesh found.');
  }
  const positions = readAccessor(json, binary, primitive.attributes.POSITION, true);
  const indices = primitive.indices === undefined ? null : readAccessor(json, binary, primitive.indices, false);
  const count = indices?.count ?? positions.count;
  if (count % 3 !== 0 || count < 3 || count / 3 > MAX_TRIANGLES) {
    throw new Error('GLB triangle budget exceeded or unsupported topology.');
  }
  const vertexData = new Float32Array(count * 3);
  const normalData = new Float32Array(count * 3);
  const min = [Infinity, Infinity, Infinity];
  const max = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < count; i++) {
    const index = indices ? indices.get(i, 0) : i;
    if (!integerInRange(index, 0, positions.count - 1)) {
      throw new Error('GLB index references a missing vertex.');
    }
    for (let j = 0; j < 3; j++) {
      const n = positions.get(index, j);
      if (!Number.isFinite(n) || Math.abs(n) > 1e9) throw new Error('Invalid GLB vertex coordinate.');
      vertexData[i * 3 + j] = n;
      min[j] = Math.min(min[j], n);
      max[j] = Math.max(max[j], n);
    }
  }
  for (let i = 0; i < count; i += 3) {
    const a = i * 3, b = a + 3, c = a + 6;
    const ux = vertexData[b] - vertexData[a], uy = vertexData[b + 1] - vertexData[a + 1], uz = vertexData[b + 2] - vertexData[a + 2];
    const vx = vertexData[c] - vertexData[a], vy = vertexData[c + 1] - vertexData[a + 1], vz = vertexData[c + 2] - vertexData[a + 2];
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const magnitude = Math.hypot(nx, ny, nz) || 1;
    nx /= magnitude; ny /= magnitude; nz /= magnitude;
    for (let vertex = i; vertex < i + 3; vertex++) {
      normalData[vertex * 3] = nx;
      normalData[vertex * 3 + 1] = ny;
      normalData[vertex * 3 + 2] = nz;
    }
  }
  const extent = Math.max(max[0] - min[0], max[1] - min[1], max[2] - min[2], 0.000001);
  return {
    positions: vertexData,
    normals: normalData,
    center: [(min[0] + max[0]) / 2, (min[1] + max[1]) / 2, (min[2] + max[2]) / 2],
    extent,
    triangles: count / 3,
  };
}

export async function fetchBoundedGlb(url: string, signal: AbortSignal): Promise<ArrayBuffer> {
  // Gallery must supply a same-origin /outputs URL; no cross-origin loads.
  if (!url.startsWith('/outputs/') || url.startsWith('//')) {
    throw new Error('GLB preview source must be a local output.');
  }
  const response = await fetch(url, { signal, credentials: 'same-origin' });
  if (!response.ok || !response.body) throw new Error('GLB preview could not be opened.');
  const expected = Number(response.headers.get('content-length') || 0);
  if (expected > MAX_PREVIEW_GLB_BYTES) {
    await response.body.cancel();
    throw new Error('GLB preview is limited to 24 MiB.');
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let length = 0;
  try {
    while (true) {
      const item = await reader.read();
      if (item.done) break;
      length += item.value.byteLength;
      if (length > MAX_PREVIEW_GLB_BYTES) throw new Error('GLB preview is limited to 24 MiB.');
      chunks.push(item.value);
    }
  } finally {
    reader.releaseLock();
    if (length > MAX_PREVIEW_GLB_BYTES) await response.body.cancel();
  }
  const result = new Uint8Array(length);
  let offset = 0;
  for (const part of chunks) {
    result.set(part, offset);
    offset += part.byteLength;
  }
  return result.buffer;
}
