import assert from 'node:assert/strict';
import { mkdirSync } from 'node:fs';
import { chromium } from 'playwright';

const base = process.env.NEXUS_SMOKE_URL || 'http://127.0.0.1:3000';
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 940 }, acceptDownloads: true });
const runtimeErrors = [];
let galleryTestPages = false;
let thumbRequests = 0;
let videoPosterRequests = 0;
page.on('pageerror', error => runtimeErrors.push(error.message));

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
  else if (path === '/api/loras' || path === '/api/gallery' || path === '/api/workflows') result = [];
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
  await page.getByText(/compatibility is unverified|No imported workflow nodes available/).waitFor();
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
  await page.getByRole('combobox', { name: 'Filter gallery media type' }).selectOption('video');
  await page.locator('.gallery-card').filter({ hasText: 'Demo clip.mp4' }).waitFor();
  await page.locator('img[alt="Video preview: Demo clip.mp4"]').waitFor();
  assert.ok(videoPosterRequests >= 1, 'Gallery must request a video poster on demand');
  assert.deepEqual(runtimeErrors, []);
  console.log('M2 browser shell: Studio, Civitai, settings persistence, import/export and Gallery passed');
} finally {
  await browser.close();
}
