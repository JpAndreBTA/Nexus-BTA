import { Link, Outlet, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import { Activity, GalleryHorizontalEnd, GitBranch, ImageUp, Images, MonitorCog, Settings, Sparkles } from 'lucide-react';

import { ExtrasPage } from '../views/extras/ExtrasPage';
import { GalleryPage } from '../views/gallery/GalleryPage';
import { HomePage } from '../views/home/HomePage';
import { ModelsPage } from '../views/modelCatalog/ModelsPage';
import { SettingsPage } from '../views/settings/SettingsPage';
import { WorkflowPage } from '../views/workflow/WorkflowPage';

function RootLayout() {
  return (
    <div className="app-shell">
      <aside className="activity-rail" aria-label="Nexus navigation">
        <div className="brand-mark">
          <Sparkles size={18} />
        </div>
        <nav className="activity-nav">
          <Link to="/" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <Activity size={18} />
            <span>Studio</span>
          </Link>
          <Link to="/extras" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <ImageUp size={18} />
            <span>Extras</span>
          </Link>
          <Link to="/models" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <MonitorCog size={18} />
            <span>Models</span>
          </Link>
          <Link to="/gallery" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <Images size={18} />
            <span>Gallery</span>
          </Link>
          <Link to="/workflow" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <GitBranch size={18} />
            <span>Workflow</span>
          </Link>
          <Link to="/settings" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <Settings size={18} />
            <span>Settings</span>
          </Link>
          <a className="nav-item" href="/ui">
            <GalleryHorizontalEnd size={18} />
            <span>Legacy</span>
          </a>
        </nav>
      </aside>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  );
}

const rootRoute = createRootRoute({
  component: RootLayout,
});

const homeRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/',
  component: HomePage,
});

const extrasRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/extras',
  component: ExtrasPage,
});

const modelsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/models',
  component: ModelsPage,
});

const galleryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/gallery',
  component: GalleryPage,
});

const workflowRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/workflow',
  component: WorkflowPage,
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/settings',
  component: SettingsPage,
});

const routeTree = rootRoute.addChildren([homeRoute, extrasRoute, modelsRoute, galleryRoute, workflowRoute, settingsRoute]);

const routerBasepath = typeof window !== 'undefined' && window.location.pathname.startsWith('/app') ? '/app' : '';

export const router = createRouter({ routeTree, basepath: routerBasepath });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
