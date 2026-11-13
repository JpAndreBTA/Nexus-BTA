import { lazy, Suspense, useEffect } from 'react';
import { Link, Outlet, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import { Activity, GalleryHorizontalEnd, GitBranch, ImageUp, Images, MonitorCog, PanelLeftClose, PanelLeftOpen, Settings } from 'lucide-react';

import { AppTopbar } from '../components/AppTopbar';

const ExtrasPage = lazy(() => import('../views/extras/ExtrasPage').then((module) => ({ default: module.ExtrasPage })));
const GalleryPage = lazy(() => import('../views/gallery/GalleryPage').then((module) => ({ default: module.GalleryPage })));
const HomePage = lazy(() => import('../views/home/HomePage').then((module) => ({ default: module.HomePage })));
const ModelsPage = lazy(() => import('../views/modelCatalog/ModelsPage').then((module) => ({ default: module.ModelsPage })));
const SettingsPage = lazy(() => import('../views/settings/SettingsPage').then((module) => ({ default: module.SettingsPage })));
const WorkflowPage = lazy(() => import('../views/workflow/WorkflowPage').then((module) => ({ default: module.WorkflowPage })));
import { useUiStore } from '../stores/uiStore';

function RootLayout() {
  const ui = useUiStore();
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.nexusTheme = ui.theme;
    root.dataset.nexusAccent = ui.accent;
    root.dataset.nexusDensity = ui.density;
    root.dataset.nexusReducedMotion = String(ui.reducedMotion);
  }, [ui.theme, ui.accent, ui.density, ui.reducedMotion]);

  return (
    <div className={ui.sidebarCollapsed ? 'app-shell rail-collapsed' : 'app-shell'}>
      <aside className="activity-rail" aria-label="Nexus navigation">
        <button className="brand-mark" type="button" onClick={() => ui.toggleSidebar()} title={ui.sidebarCollapsed ? 'Show navigation' : 'Hide navigation'}>
          <strong>N</strong>
          {ui.sidebarCollapsed ? <PanelLeftOpen size={12} /> : <PanelLeftClose size={12} />}
        </button>
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
          <a className="nav-item" href="/ui">
            <GalleryHorizontalEnd size={18} />
            <span>Legacy</span>
          </a>
        </nav>
        <nav className="activity-nav rail-bottom">
          <Link to="/settings" activeProps={{ className: 'nav-item active' }} inactiveProps={{ className: 'nav-item' }}>
            <Settings size={18} />
            <span>Settings</span>
          </Link>
        </nav>
      </aside>
      <main className="app-main">
        <AppTopbar />
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
  component: () => <Suspense fallback={<div className="page workspace-loading" role="status">Loading workspace…</div>}><HomePage /></Suspense>,
});

const extrasRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/extras',
  component: () => <Suspense fallback={<div className="page workspace-loading" role="status">Loading workspace…</div>}><ExtrasPage /></Suspense>,
});

const modelsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/models',
  component: () => <Suspense fallback={<div className="page workspace-loading" role="status">Loading workspace…</div>}><ModelsPage /></Suspense>,
});

const galleryRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/gallery',
  component: () => <Suspense fallback={<div className="page workspace-loading" role="status">Loading workspace…</div>}><GalleryPage /></Suspense>,
});

const workflowRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/workflow',
  component: () => <Suspense fallback={<div className="page workspace-loading" role="status">Loading workspace…</div>}><WorkflowPage /></Suspense>,
});

const settingsRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: '/settings',
  component: () => <Suspense fallback={<div className="page workspace-loading" role="status">Loading workspace…</div>}><SettingsPage /></Suspense>,
});

const routeTree = rootRoute.addChildren([homeRoute, extrasRoute, modelsRoute, galleryRoute, workflowRoute, settingsRoute]);

const routerBasepath = typeof window !== 'undefined' && window.location.pathname.startsWith('/app') ? '/app' : '';

export const router = createRouter({ routeTree, basepath: routerBasepath });

declare module '@tanstack/react-router' {
  interface Register {
    router: typeof router;
  }
}
