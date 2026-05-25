import { Link, Outlet, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import { Activity, Download, GalleryHorizontalEnd, GitBranch, ImageUp, Images, MonitorCog, PanelLeftClose, PanelLeftOpen, RefreshCw, Settings } from 'lucide-react';

import { ExtrasPage } from '../views/extras/ExtrasPage';
import { GalleryPage } from '../views/gallery/GalleryPage';
import { HomePage } from '../views/home/HomePage';
import { ModelsPage } from '../views/modelCatalog/ModelsPage';
import { SettingsPage } from '../views/settings/SettingsPage';
import { WorkflowPage } from '../views/workflow/WorkflowPage';
import { useModelCatalogQuery } from '../api/queries';
import { useUiStore } from '../stores/uiStore';
import { useGenerationStore } from '../stores/generationStore';

const shellPresets = ['SD', 'SDXL', 'Flux', 'Qwen', 'ZImageTurbo', 'Lumina', 'Wan', 'LTX', 'Anima'];

function shellPresetLabel(preset: string) {
  if (preset === 'SD') return 'SD 1.5';
  if (preset === 'Wan') return 'WAN 2.2';
  if (preset === 'LTX') return 'LTX 2.3';
  return preset === 'ZImageTurbo' ? 'Z-IMG' : preset.toUpperCase();
}

function checkpointOptions(catalog: ReturnType<typeof useModelCatalogQuery>['data'], preset: string) {
  const lowerPreset = preset.toLowerCase();
  const rules: Record<string, string[]> = {
    sd: ['sd15', 'sd1', '1.5', 'dreamshaper'],
    sdxl: ['sdxl', 'xl', 'illustrious'],
    flux: ['flux'],
    qwen: ['qwen'],
    zimageturbo: ['z_image', 'zimage', 'z-image'],
    lumina: ['lumina'],
    wan: ['wan'],
    ltx: ['ltx'],
    anima: ['anima'],
  };
  const assets = [...(catalog?.categories.checkpoints ?? []), ...(catalog?.categories.unet ?? []), ...(catalog?.categories.diffusion_models ?? [])];
  const tokens = rules[lowerPreset] || [lowerPreset];
  const matches = assets.filter((asset) => {
    const haystack = `${asset.name} ${asset.relative_path} ${asset.folder} ${asset.tags?.join(' ')}`.toLowerCase();
    return tokens.some((token) => haystack.includes(token));
  });
  return (matches.length ? matches : assets).slice(0, 80).map((asset) => ({
    label: asset.relative_path || asset.name,
    value: asset.relative_path || asset.path,
    name: asset.name,
  }));
}

function AppTopbar() {
  const catalog = useModelCatalogQuery();
  const generation = useGenerationStore();
  const checkpoints = checkpointOptions(catalog.data, generation.preset);

  return (
    <header className="app-topbar">
      <div className="model-strip" aria-label="Base model selector">
        <span>Base Models:</span>
        <div className="model-chip-row">
          {shellPresets.map((preset) => (
            <button className={generation.preset === preset ? 'active' : ''} type="button" key={preset} onClick={() => generation.setPreset(preset)}>
              {shellPresetLabel(preset)}
            </button>
          ))}
        </div>
      </div>
      <div className="checkpoint-selector">
        <span>Checkpoint:</span>
        <select
          value={generation.modelPath}
          onChange={(event) => {
            const selected = checkpoints.find((checkpoint) => checkpoint.value === event.currentTarget.value);
            generation.setModel(event.currentTarget.value, selected?.name || '');
          }}
        >
          <option value="">Automatic</option>
          {checkpoints.map((checkpoint) => (
            <option key={checkpoint.value} value={checkpoint.value}>
              {checkpoint.label}
            </option>
          ))}
        </select>
        <button className="icon-button" type="button" onClick={() => catalog.refetch()} title="Refresh model catalog">
          <RefreshCw size={13} />
        </button>
      </div>
      <div className="topbar-actions">
        <Link className="flat-button compact-action" to="/models">
          <Download size={13} />
          Civitai
        </Link>
      </div>
    </header>
  );
}

function RootLayout() {
  const ui = useUiStore();

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
