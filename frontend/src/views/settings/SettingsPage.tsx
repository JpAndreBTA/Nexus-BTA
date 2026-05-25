import { Globe2, HardDrive, Network, Server, ShieldAlert } from 'lucide-react';

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
  const origin = typeof window !== 'undefined' ? window.location.origin : '';

  return (
    <section className="page settings-layout">
      <span className="route-sentinel" data-route="settings" aria-label="Settings workspace" />

      <div className="settings-grid">
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
