import assert from 'node:assert/strict';
import { chromium } from 'playwright';

const base = process.env.NEXUS_SMOKE_URL || 'http://127.0.0.1:3000';
const browser = await chromium.launch({ headless: true });
const page = await browser.newPage({ viewport: { width: 1600, height: 940 }, acceptDownloads: true });
const runtimeErrors = [];
page.on('pageerror', error => runtimeErrors.push(error.message));

await page.route('**/api/**', async (route) => {
  const url = new URL(route.request().url());
  const path = url.pathname;
  let result = {};
  if (path === '/api/models') result = { categories: { checkpoints: [], unet: [], diffusion_models: [], loras: [], vae: [], clip: [], controlnet: [] } };
  else if (path === '/api/loras' || path === '/api/gallery' || path === '/api/workflows') result = [];
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
  await page.getByRole('button', { name: 'Civitai' }).click();
  await page.getByRole('dialog', { name: 'Nexus Civitai downloader' }).waitFor();
  await page.getByPlaceholder('Search terms or paste a Civitai URL...').fill('test model');
  await page.getByRole('button', { name: 'Resolve URL' }).click();
  await page.getByText('QA Mock Model').waitFor();
  await page.keyboard.press('Escape');
  assert.equal(await page.getByRole('dialog', { name: 'Nexus Civitai downloader' }).count(), 0);
  assert.equal(await page.evaluate(() => document.body.style.overflow), '');

  await page.goto(`${base}/settings`, { waitUntil: 'domcontentloaded' });
  await page.getByText('Customize your workspace').waitFor();
  await page.getByLabel('Theme').selectOption('midnight');
  await page.getByLabel('Accent').selectOption('violet');
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
  assert.deepEqual(runtimeErrors, []);
  console.log('M2 browser shell: Studio, Civitai, settings persistence, import/export and Gallery passed');
} finally {
  await browser.close();
}
