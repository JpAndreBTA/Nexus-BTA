import { Server, ShieldCheck } from 'lucide-react';

import { useHealthQuery } from '../../api/queries';

export function HomePage() {
  const health = useHealthQuery();

  return (
    <section className="page page-grid">
      <header className="page-header">
        <div>
          <p className="eyebrow">Nexus BTA Web App</p>
          <h1>Studio</h1>
        </div>
        <span className={health.data?.nexus === 'ok' ? 'status-pill ok' : 'status-pill'}>
          <Server size={14} />
          {health.data?.nexus === 'ok' ? 'Backend online' : 'Backend check'}
        </span>
      </header>

      <div className="surface hero-surface">
        <div>
          <p className="eyebrow">Migration track</p>
          <h2>React shell active, legacy UI preserved.</h2>
          <p className="muted">
            This shell is the migration base for the official responsive web app. The current legacy interface remains available at /ui while each tool is rebuilt in React.
          </p>
        </div>
        <ShieldCheck className="hero-icon" size={72} />
      </div>

      <div className="panel-grid">
        <article className="surface">
          <h3>Next slice</h3>
          <p className="muted">Extras will be migrated first because it already has isolated API jobs, uploads, polling and previews.</p>
        </article>
        <article className="surface">
          <h3>Responsive target</h3>
          <p className="muted">Desktop uses rail + panels. Tablet and mobile collapse into scrollable tool sections with stable controls.</p>
        </article>
      </div>
    </section>
  );
}
