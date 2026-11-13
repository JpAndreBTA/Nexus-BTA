import { useRef, useState } from 'react';
import { Download, Globe2, HardDrive, Network, Palette, RotateCcw, Server, ShieldAlert, Upload } from 'lucide-react';
import { useUiStore, type NexusTheme, type NexusAccent, type ShellDensity } from '../../stores/uiStore';

import { useHealthQuery } from '../../api/queries';
import { exportWorkspacePreset, importWorkspacePreset, MAX_PRESET_BYTES } from '../../shared/workspacePreset';

const checklist = [
  'Keep local mode on 127.0.0.1 for personal use.',
  'LAN is disabled by default; an authenticated gateway is still required.',
  'Do not expose StartTunnel or the backend publicly without authentication and HTTPS.',
  'Restrict CORS and file endpoints before remote production.',
  'Keep /ui legacy available until /app reaches feature parity.',
];

export function SettingsPage() {
  const health = useHealthQuery();
  const ui = useUiStore();
  const importFileRef = useRef<HTMLInputElement | null>(null);
  const [layoutMessage, setLayoutMessage] = useState('');
  function downloadLayout() {
    const contents = exportWorkspacePreset(useUiStore.getState());
    const url = URL.createObjectURL(new Blob([contents], { type: 'application/json' }));
    const link = document.createElement('a');
    link.href = url;
    link.download = 'nexus-workspace-preset.json';
    document.body.append(link);
    link.click();
    link.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    setLayoutMessage('Workspace preset exported (appearance and layout only).');
  }
  async function uploadLayout(file: File | undefined) {
    if (!file) return;
    try {
      if (file.size > MAX_PRESET_BYTES) throw new Error('Workspace preset exceeds 16 KB.');
      const preferences = importWorkspacePreset(await file.text());
      useUiStore.setState(preferences);
      setLayoutMessage('Workspace preset imported successfully.');
    } catch (error) {
      setLayoutMessage(error instanceof Error ? error.message : 'Unable to import preset.');
    } finally {
      if (importFileRef.current) importFileRef.current.value = '';
    }
  }
  const origin = typeof window !== 'undefined' ? window.location.origin : '';

  return (
    <section className="page settings-layout">
      <span className="route-sentinel" data-route="settings" aria-label="Settings workspace" />

      <div className="settings-grid">
        <article className="surface settings-card wide-card appearance-workspace">
          <header className="appearance-heading"><div><Palette size={22} /><h2>Customize your workspace</h2></div>
            <button type="button" className="mini-button" onClick={ui.resetAppearance}><RotateCcw size={14} /> Reset appearance</button>
          </header>
          <p className="muted">Live preview. Appearance preferences are stored only in this browser.</p>
          <div className="appearance-controls">
            <label>Theme
              <select value={ui.theme} onChange={(event) => ui.setTheme(event.currentTarget.value as NexusTheme)}>
                <option value="graphite">Graphite</option><option value="midnight">Midnight</option><option value="obsidian">Obsidian</option>
              </select>
            </label>
            <label>Accent
              <select value={ui.accent} onChange={(event) => ui.setAccent(event.currentTarget.value as NexusAccent)}>
                <option value="red">Red</option><option value="blue">Blue</option><option value="violet">Violet</option>
                <option value="emerald">Emerald</option><option value="amber">Amber</option>
              </select>
            </label>
            <label>Density
              <select value={ui.density} onChange={(event) => ui.setDensity(event.currentTarget.value as ShellDensity)}>
                <option value="compact">Compact</option><option value="comfortable">Comfortable</option>
              </select>
            </label>
            <label className="appearance-check"><input type="checkbox" checked={ui.reducedMotion}
              onChange={(event) => ui.setReducedMotion(event.currentTarget.checked)} /> Reduce animation</label>
          </div>
          <div className="appearance-controls" role="group" aria-label="Studio panel width">
            <label>Studio controls: {ui.controlsWidth}px
              <input type="range" min={320} max={600} step={10} value={ui.controlsWidth}
                onChange={(event) => ui.setControlsWidth(Number(event.currentTarget.value))} />
            </label>
            <label>Gallery panel: {ui.galleryWidth}px
              <input type="range" min={220} max={480} step={10} value={ui.galleryWidth}
                onChange={(event) => ui.setGalleryWidth(Number(event.currentTarget.value))} />
            </label>
          </div>
          <div className="appearance-preset-actions">
            <button type="button" className="mini-button" onClick={downloadLayout}><Download size={14} /> Export layout JSON</button>
            <button type="button" className="mini-button" onClick={() => importFileRef.current?.click()}><Upload size={14} /> Import layout JSON</button>
            <input ref={importFileRef} type="file" accept=".json,application/json" hidden aria-label="Import Nexus workspace preset"
              onChange={(event) => { void uploadLayout(event.currentTarget.files?.[0]); }} />
          </div>
          {layoutMessage && <p className="muted" role="status" aria-live="polite">{layoutMessage}</p>}
          <div className="appearance-preview" aria-label="Theme preview">
            <span>Preview: model panel</span><button type="button" className="primary-button" onClick={() => ui.toggleSidebar()}>
              {ui.sidebarCollapsed ? 'Expand navigation' : 'Collapse navigation'}
            </button>
          </div>
        </article>
        <article className="surface settings-card">
          <Server size={22} />
          <h2>Backend</h2>
          <dl>
            <div>
              <dt>Current UI</dt>
              <dd>{origin || '-'}</dd>
            </div>
            <div>
              <dt>Comfy URL</dt>
              <dd>{health.data?.comfy_url || '-'}</dd>
            </div>
            <div>
              <dt>Comfy Running</dt>
              <dd>{health.data?.comfy_running ? 'yes' : 'on demand'}</dd>
            </div>
          </dl>
        </article>

        <article className="surface settings-card">
          <HardDrive size={22} />
          <h2>Paths</h2>
          <dl>
            <div>
              <dt>Models</dt>
              <dd>{health.data?.models_dir || '-'}</dd>
            </div>
            <div>
              <dt>Custom Nodes</dt>
              <dd>{health.data?.custom_nodes_dir || '-'}</dd>
            </div>
          </dl>
        </article>

        <article className="surface settings-card">
          <Network size={22} />
          <h2>Access Modes</h2>
          <div className="access-list">
            <span>Local: 127.0.0.1</span>
            <span>LAN: 0.0.0.0 with trusted network only</span>
            <span>Internet: reverse proxy, HTTPS and auth required</span>
          </div>
        </article>

        <article className="surface settings-card warning-card">
          <ShieldAlert size={22} />
          <h2>Security Gate</h2>
          <ul>
            {checklist.map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </article>

        <article className="surface settings-card wide-card">
          <Globe2 size={22} />
          <h2>Migration Status</h2>
          <p className="muted">
            React `/app` is now available beside the legacy `/ui`. The next official-release work is feature parity, remote safety controls and final smoke coverage.
          </p>
        </article>
      </div>
    </section>
  );
}
