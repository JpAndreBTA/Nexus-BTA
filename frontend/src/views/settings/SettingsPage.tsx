import { Globe2, HardDrive, Network, Palette, RotateCcw, Server, ShieldAlert } from 'lucide-react';
import { useUiStore, type NexusTheme, type NexusAccent, type ShellDensity } from '../../stores/uiStore';

import { useHealthQuery } from '../../api/queries';

const checklist = [
  'Keep local mode on 127.0.0.1 for personal use.',
  'Use 0.0.0.0 only for trusted LAN testing.',
  'Require auth and HTTPS before exposing to the internet.',
  'Restrict CORS and file endpoints before remote production.',
  'Keep /ui legacy available until /app reaches feature parity.',
];

export function SettingsPage() {
  const health = useHealthQuery();
  const ui = useUiStore();
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
