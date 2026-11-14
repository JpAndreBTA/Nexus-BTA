import assert from 'node:assert/strict';
import test from 'node:test';
import { exportWorkspacePreset, importWorkspacePreset } from '../src/shared/workspacePreset.ts';

const sample = {
  theme: 'midnight', accent: 'blue', density: 'comfortable', reducedMotion: true,
  controlsWidth: 480, galleryWidth: 320, sidebarCollapsed: false,
  studioControlsCollapsed: false, studioGalleryOpen: true, studioGalleryExpanded: false,
};

test('workspace roundtrip retains only allowed layout options', () => {
  const json = exportWorkspacePreset({ ...sample, prompt: 'PRIVATE', modelsDir: 'C:\\secret' });
  assert.deepEqual(importWorkspacePreset(json), sample);
  assert.equal(json.includes('PRIVATE'), false);
  assert.equal(json.includes('secret'), false);
});

test('invalid schemas, limits and prototype injection do not enter user state', () => {
  assert.throws(() => importWorkspacePreset('{invalid'), /not valid JSON/);
  assert.throws(() => importWorkspacePreset('x'.repeat(20000)), /exceeds/);
  assert.throws(() => importWorkspacePreset(JSON.stringify({
    format: 'nexus-bta-workspace', version: 9, preferences: sample,
  })), /Unsupported/);
  const poisoned = JSON.stringify({ format: 'nexus-bta-workspace', version: 1, preferences: { ...sample, theme: 'url(javascript:bad)', __proto__: { admin: true } } });
  assert.throws(() => importWorkspacePreset(poisoned), /Invalid theme/);
});

test('pixel widths out of range are rejected rather than used as CSS', () => {
  const json = JSON.stringify({ format: 'nexus-bta-workspace', version: 1, preferences: { ...sample, galleryWidth: 10000 } });
  assert.throws(() => importWorkspacePreset(json), /galleryWidth/);
});
