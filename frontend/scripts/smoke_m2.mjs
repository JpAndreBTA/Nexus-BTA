import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';

const base = process.env.NEXUS_SMOKE_URL || 'http://127.0.0.1:3000';
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 940 }, acceptDownloads: true });

function glbTriangleFixture() {
  const positions = new Float32Array([0, 0, 0, 1, 0, 0, 0, 1, 0]);
  const meta = {
    asset: { version: '2.0' },
    buffers: [{ byteLength: positions.byteLength }],
    bufferViews: [{ buffer: 0, byteOffset: 0, byteLength: positions.byteLength }],
    accessors: [{ bufferView: 0, componentType: 5126, count: 3, type: 'VEC3' }],
    meshes: [{ primitives: [{ attributes: { POSITION: 0 }, mode: 4 }] }],
  };
  const json = Buffer.from(JSON.stringify(meta));
  const padded = Buffer.concat([json, Buffer.alloc((4 - (json.length % 4)) % 4, 32)]);
  const bin = Buffer.from(positions.buffer);
  const glb = Buffer.alloc(12 + 8 + padded.length + 8 + bin.length);
  glb.writeUInt32LE(0x46546c67, 0);
  glb.writeUInt32LE(2, 4);
  glb.writeUInt32LE(glb.length, 8);
  glb.writeUInt32LE(padded.length, 12);
  glb.writeUInt32LE(0x4e4f534a, 16);
  padded.copy(glb, 20);
  const offset = 20 + padded.length;
  glb.writeUInt32LE(bin.length, offset);
  glb.writeUInt32LE(0x004e4942, offset + 4);
  bin.copy(glb, offset + 8);
  return glb;
}

const runtimeErrors = [];
let galleryTestPages = false;
let thumbRequests = 0;
let videoPosterRequests = 0;
let selectedWorkflowCapabilityRequests = 0;
page.on('pageerror', error => runtimeErrors.push(error.message));

await page.route('**/outputs/**', async (route) => {
  const path = new URL(route.request().url()).pathname;
  if (path.endsWith('.glb')) {
    await route.fulfill({ status: 200, contentType: 'model/gltf-binary', body: glbTriangleFixture() });
    return;
  }
  await route.continue();
});

await page.route('**/api/**', async (route) => {
  const url = new URL(route.request().url());
  const path = url.pathname;
  let result = {};
  if (path.startsWith('/api/gallery/video-thumb/')) {
    videoPosterRequests++;
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
    await route.fulfill({ status: 200, contentType: 'image/png', body: png });
    return;
  }
  if (path.startsWith('/api/gallery/thumb/')) {
    thumbRequests++;
    // Tiny opaque fixture: browser exercises image decoding without opening local files.
    const png = Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64');
    await route.fulfill({ status: 200, contentType: 'image/png', body: png });
    return;
  }
  if (path === '/api/models') result = { categories: { checkpoints: [], unet: [], diffusion_models: [], loras: [], vae: [], clip: [], controlnet: [] } };
  else if (path === '/api/gallery/page') {
    const cursor = url.searchParams.get('cursor');
    const q = url.searchParams.get('q');
    const mediaType = url.searchParams.get('media');
    const media = (id, name = `Acceptance ${id}.png`, kind = 'image') => ({
      title: name, filename: name, path: name, relative_path: name,
      folder: '', image: `/outputs/${encodeURIComponent(name)}`,
      thumb: kind === 'image'
        ? `/api/gallery/thumb/${encodeURIComponent(name)}?size=320`
        : kind === 'video'
          ? `/api/gallery/video-thumb/${encodeURIComponent(name)}?size=320`
          : '',
      media_type: kind, modified: id, model: 'QA model', preset: 'SD',
    });
    result = !galleryTestPages
      ? { items: [], total: 0, next_cursor: null }
      : mediaType === '3d'
        ? { items: [media(5, 'Forest model.glb', '3d')], total: 1, next_cursor: null }
        : mediaType === 'video'
          ? { items: [media(6, 'Demo clip.mp4', 'video')], total: 1, next_cursor: null }
        : q === 'Ocean'
          ? { items: [media(3, 'Ocean coast.png')], total: 1, next_cursor: null }
          : cursor
            ? { items: [media(1)], total: 2, next_cursor: null }
            : { items: [media(2)], total: 2, next_cursor: 'next-page' };
  }
  else if (path === '/api/workflows') result = [{
    id: 'qa-workflow', name: 'QA Mock Workflow', path: '/mock/qa.json',
    format: 'api', node_count: 2, class_types: ['KSampler', 'VAEDecode'], tags: [],
  }];
  else if (path === '/api/workflows/qa-workflow/analysis') result = {
    workflow: { id: 'qa-workflow', name: 'QA Mock Workflow', path: '/mock/qa.json',
      format: 'api', node_count: 2, class_types: ['KSampler', 'VAEDecode'], tags: [] },
    missing_nodes: [], available_nodes: 0, manager_suggestions: {},
    dependency_targets: [], dependencies_installed: [], dependency_errors: {},
    visual_graph: { nodes: [], links: [], width: 700, height: 500 },
    workflow_settings: {},
  };
  else if (path === '/api/loras' || path === '/api/gallery') result = [];
  else if (path === '/api/comfy/workflow-inputs') {
    assert.equal(url.searchParams.get('workflow_id'), 'qa-workflow');
    result = {
      workflow_id: 'qa-workflow', status: 'offline', runtime_checked: false, static_valid: false,
      ready: false, node_count: 2, registered_count: 0, missing_classes: 0, broken_links: 0,
      missing_required_inputs: 0, choice_mismatches: 0, unverified_inputs: 0,
      findings: [], truncated_findings: false,
      note: 'Saved workflow inspection only; no Comfy execution.',
    };
  }
  else if (path === '/api/comfy/workflow-capabilities') {
    selectedWorkflowCapabilityRequests++;
    const workflow_id = url.searchParams.get('workflow_id');
    assert.equal(workflow_id, 'qa-workflow');
    result = {
      status: 'offline', runtime_checked: false, ready: false,
      requested: ['KSampler', 'VAEDecode'], available: [], missing: [],
      unverified: ['KSampler', 'VAEDecode'], registered_count: 0,
      workflow_id, required_count: 2, invalid_class_count: 0,
    };
  }
  else if (path === '/api/comfy/capabilities') {
    const requested = (url.searchParams.get('classes') || '').split(',').filter(Boolean);
    result = {
      status: 'offline', runtime_checked: false, ready: false,
      requested, available: [], missing: [], unverified: requested, registered_count: 0,
    };
  }
  else if (path === '/api/health') result = {
    comfy_running: false,
    comfy_url: 'http://127.0.0.1:8189',
    models_dir: '/mock/models',
    custom_nodes_dir: '/mock/custom_nodes',
  };
  else if (path === '/api/civitai/search') result = {
    items: [{ id: 42, name: 'QA Mock Model', type: 'Checkpoint', versions: [{ url: 'https://civitai.com/models/42', base_model: 'SD 1.5' }] }],
  };
  else if (path === '/api/model-tree') result = { ok: true };
  await route.fulfill({ status: 200, contentType: 'application/json', body: JSON.stringify(result) });
});

try {
  await page.goto(base, { waitUntil: 'domcontentloaded' });
  await page.locator('[data-route="studio"]').waitFor({ state: 'attached', timeout: 15000 });
  await page.getByRole('button', { name: /Node Workflow/ }).click();
  await page.getByText(/ComfyUI is offline. Node compatibility is unverified/).waitFor();
  await page.getByText('Input ABI unverified: ComfyUI is offline.').waitFor();
  assert.ok(selectedWorkflowCapabilityRequests >= 1, 'Studio should check exact workflow classes from backend');
  await page.getByRole('button', { name: 'Linear Viewer' }).click();
  await page.getByRole('button', { name: 'Civitai' }).click();
  await page.getByRole('dialog', { name: 'Nexus Civitai downloader' }).waitFor();
  await page.getByPlaceholder('Search terms or paste a Civitai URL...').fill('test model');
  await page.getByRole('button', { name: 'Resolve URL' }).click();
  await page.getByText('QA Mock Model').waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await page.getByRole('dialog', { name: 'Nexus Civitai downloader' }).count(), 0);
  assert.equal(await page.evaluate(() => document.body.style.overflow), '');

  // Inpaint's reference and mask must scale/pan as one viewport, not drift apart.
  await page.locator('.reference-source-block input[type=file]').first().setInputFiles({
    name: 'qa-reference.png', mimeType: 'image/png',
    buffer: Buffer.from('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=', 'base64'),
  });
  await page.getByRole('button', { name: 'Inpaint Canvas' }).click();
  await page.getByRole('button', { name: 'Zoom in inpaint image' }).click();
  await page.getByText('125%', { exact: true }).waitFor();
  const sharedViewport = page.locator('.inpaint-viewport');
  assert.equal(await sharedViewport.locator('img').count(), 1);
  assert.equal(await sharedViewport.locator('canvas').count(), 1);
  assert.ok((await sharedViewport.evaluate((element) => element.style.transform)).includes('scale(1.25)'));

  // A 2048x1024 source is edited on the 1536px mask grid, but its emitted
  // opaque binary PNG must map back to the full native source resolution.
  const nativeSource = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 2048;
    canvas.height = 1024;
    const context = canvas.getContext('2d');
    context.fillStyle = '#6f91a3';
    context.fillRect(0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/png');
  });
  await page.locator('.reference-source-block input[type=file]').first().setInputFiles({
    name: 'qa-native-reference.png', mimeType: 'image/png',
    buffer: Buffer.from(nativeSource.split(',')[1], 'base64'),
  });
  await page.getByText(/Source 2048×1024 · edit 1536×768 · mask export 2048×1024/).waitFor();
  await page.evaluate(() => {
    const original = HTMLCanvasElement.prototype.toDataURL;
    window.__nexusMaskExportSizes = [];
    HTMLCanvasElement.prototype.toDataURL = function (...args) {
      const exported = original.apply(this, args);
      if (args[0] === 'image/png' && this.width > 1536) {
        window.__nexusMaskExportSizes.push([this.width, this.height]);
        window.__nexusMaskExportPng = exported;
      }
      return exported;
    };
  });
  const maskViewport = page.locator('canvas[aria-label="Inpaint mask canvas"]');
  await maskViewport.scrollIntoViewIfNeeded();
  const bounds = await maskViewport.boundingBox();
  assert.ok(bounds && bounds.width > 30 && bounds.height > 30, 'mask editor must have measurable bounds');
  await page.mouse.move(bounds.x + bounds.width / 2, bounds.y + bounds.height / 2);
  await page.mouse.down();
  await page.mouse.up();
  await page.waitForFunction(() => window.__nexusMaskExportSizes?.some(
    (size) => size[0] === 2048 && size[1] === 1024,
  ));
  const maskPixels = await page.evaluate(async () => {
    const source = new Image();
    source.src = window.__nexusMaskExportPng;
    await source.decode();
    const canvas = document.createElement('canvas');
    canvas.width = source.naturalWidth;
    canvas.height = source.naturalHeight;
    const context = canvas.getContext('2d', { willReadFrequently: true });
    context.drawImage(source, 0, 0);
    const center = [...context.getImageData(1024, 512, 1, 1).data];
    const corner = [...context.getImageData(0, 0, 1, 1).data];
    return { center, corner };
  });
  assert.deepEqual(maskPixels.center, [255, 255, 255, 255], 'painted center must remain white');
  assert.deepEqual(maskPixels.corner, [0, 0, 0, 255], 'untouched pixels must remain opaque black');

  await page.screenshot({ path: 'artifacts/inpaint-native-mask.png', fullPage: true });

  await page.goto(`${base}/settings`, { waitUntil: 'domcontentloaded' });
  await page.getByText('Customize your workspace').waitFor();
  await page.locator('.appearance-controls select').nth(0).selectOption('midnight');
  await page.locator('.appearance-controls select').nth(1).selectOption('violet');
  assert.equal(await page.evaluate(() => document.documentElement.dataset.nexusTheme), 'midnight');
  assert.equal(await page.evaluate(() => document.documentElement.dataset.nexusAccent), 'violet');

  const [download] = await Promise.all([
    page.waitForEvent('download'),
    page.getByRole('button', { name: 'Export layout JSON' }).click(),
  ]);
  assert.equal(download.suggestedFilename(), 'nexus-workspace-preset.json');
  const layout = {
    format: 'nexus-bta-workspace',
    version: 1,
    preferences: {
      theme: 'obsidian', accent: 'emerald', density: 'comfortable',
      reducedMotion: true, controlsWidth: 480, galleryWidth: 380,
      sidebarCollapsed: false, studioControlsCollapsed: false,
      studioGalleryOpen: true, studioGalleryExpanded: false,
    },
  };
  await page.getByLabel('Import Nexus workspace preset').setInputFiles({
    name: 'workspace.json', mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(layout), 'utf8'),
  });
  await page.getByText('Workspace preset imported successfully.').waitFor();
  assert.equal(await page.evaluate(() => document.documentElement.dataset.nexusTheme), 'obsidian');
  assert.equal(await page.evaluate(() => document.documentElement.style.getPropertyValue('--nexus-controls-width')), '480px');

  await page.goto(`${base}/gallery`, { waitUntil: 'domcontentloaded' });
  await page.locator('[data-route="gallery"]').waitFor({ state: 'attached' });
  await page.getByText('No outputs found.').waitFor();
  galleryTestPages = true;
  await page.getByRole('button', { name: 'Refresh', exact: true }).click();
  await page.getByRole('button', { name: /Load more/ }).waitFor();
  await page.getByRole('button', { name: /Load more/ }).click();
  await page.getByText('2 / 2 loaded').waitFor();
  await page.getByRole('button', { name: 'Acceptance 1.png' }).waitFor();
  mkdirSync('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/gallery-paginated.png', fullPage: true });
  assert.ok(thumbRequests >= 1, 'image grid must request thumbnail endpoint');

  await page.getByRole('textbox', { name: 'Search all gallery outputs' }).fill('Ocean');
  await page.locator('.gallery-card').filter({ hasText: 'Ocean coast.png' }).waitFor();
  await page.getByText('1 / 1 loaded').waitFor();
  await page.getByRole('textbox', { name: 'Search all gallery outputs' }).fill('');
  await page.getByRole('combobox', { name: 'Filter gallery media type' }).selectOption('3d');
  await page.locator('.gallery-card').filter({ hasText: 'Forest model.glb' }).waitFor();
  await page.getByText('1 / 1 loaded').waitFor();
  await page.getByRole('img', { name: /Experimental GLB orbit preview/ }).waitFor();
  // On supported browsers, render one real mocked triangle on a WebGL2 canvas.
  if (await page.evaluate(() => Boolean(document.createElement('canvas').getContext('webgl2')))) {
    await page.getByText(/First mesh primitive/).waitFor({ timeout: 15000 });
  } else {
    await page.getByText(/WebGL2 is unavailable/).waitFor();
  }
  await page.screenshot({ path: 'artifacts/gallery-glb-preview.png', fullPage: true });
  await page.getByRole('combobox', { name: 'Filter gallery media type' }).selectOption('video');
  await page.locator('.gallery-card').filter({ hasText: 'Demo clip.mp4' }).waitFor();
  await page.locator('img[alt="Video preview: Demo clip.mp4"]').waitFor();
  assert.ok(videoPosterRequests >= 1, 'Gallery must request a video poster on demand');
  assert.deepEqual(runtimeErrors, []);
  console.log('M2 browser shell: Studio, Civitai, settings persistence, import/export and Gallery passed');
} finally {
  await browser.close();
}
