import type { UiState, NexusAccent, NexusTheme, ShellDensity } from '../stores/uiStore';

export const WORKSPACE_PRESET_FORMAT = 'nexus-bta-workspace';
export const WORKSPACE_PRESET_VERSION = 1;
export const MAX_PRESET_BYTES = 16 * 1024;

export type WorkspacePreferences = Pick<
  UiState,
  'theme' | 'accent' | 'density' | 'reducedMotion' | 'controlsWidth' | 'galleryWidth' |
  'sidebarCollapsed' | 'studioControlsCollapsed' | 'studioGalleryOpen' | 'studioGalleryExpanded'
>;

export function exportWorkspacePreset(settings: WorkspacePreferences): string {
  const preferences: WorkspacePreferences = {
    theme: settings.theme,
    accent: settings.accent,
    density: settings.density,
    reducedMotion: settings.reducedMotion,
    controlsWidth: settings.controlsWidth,
    galleryWidth: settings.galleryWidth,
    sidebarCollapsed: settings.sidebarCollapsed,
    studioControlsCollapsed: settings.studioControlsCollapsed,
    studioGalleryOpen: settings.studioGalleryOpen,
    studioGalleryExpanded: settings.studioGalleryExpanded,
  };
  // Intentionally do not include prompts, user paths, model names, credentials,
  // generation data or browser storage keys.
  return JSON.stringify({
    format: WORKSPACE_PRESET_FORMAT,
    version: WORKSPACE_PRESET_VERSION,
    preferences,
  }, null, 2);
}

function record(value: unknown): Record<string, unknown> | null {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
    ? value as Record<string, unknown> : null;
}

function choice<T extends string>(value: unknown, allowed: readonly T[], label: string): T {
  if (typeof value === 'string' && allowed.includes(value as T)) return value as T;
  throw new Error(`Invalid ${label} in workspace preset.`);
}

function booleanField(value: unknown, label: string): boolean {
  if (typeof value === 'boolean') return value;
  throw new Error(`Invalid ${label} in workspace preset.`);
}

function widthField(value: unknown, min: number, max: number, label: string): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max) {
    throw new Error(`Invalid ${label} in workspace preset.`);
  }
  return Math.round(value);
}

export function importWorkspacePreset(json: string): WorkspacePreferences {
  if (new TextEncoder().encode(json).byteLength > MAX_PRESET_BYTES) throw new Error('Workspace preset exceeds 16 KB.');
  let decoded: unknown;
  try {
    decoded = JSON.parse(json);
  } catch {
    throw new Error('Workspace preset is not valid JSON.');
  }
  const root = record(decoded);
  if (!root || root.format !== WORKSPACE_PRESET_FORMAT || root.version !== WORKSPACE_PRESET_VERSION) {
    throw new Error('Unsupported Nexus workspace preset version.');
  }
  const settings = record(root.preferences);
  if (!settings) throw new Error('Workspace preferences are missing.');
  return {
    theme: choice<NexusTheme>(settings.theme, ['graphite', 'midnight', 'obsidian'], 'theme'),
    accent: choice<NexusAccent>(settings.accent, ['red', 'blue', 'violet', 'emerald', 'amber'], 'accent'),
    density: choice<ShellDensity>(settings.density, ['compact', 'comfortable'], 'density'),
    reducedMotion: booleanField(settings.reducedMotion, 'reducedMotion'),
    controlsWidth: widthField(settings.controlsWidth, 320, 600, 'controlsWidth'),
    galleryWidth: widthField(settings.galleryWidth, 220, 480, 'galleryWidth'),
    sidebarCollapsed: booleanField(settings.sidebarCollapsed, 'sidebarCollapsed'),
    studioControlsCollapsed: booleanField(settings.studioControlsCollapsed, 'studioControlsCollapsed'),
    studioGalleryOpen: booleanField(settings.studioGalleryOpen, 'studioGalleryOpen'),
    studioGalleryExpanded: booleanField(settings.studioGalleryExpanded, 'studioGalleryExpanded'),
  };
}
