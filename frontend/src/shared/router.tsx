import { useState } from 'react';
import { Link, Outlet, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import { Activity, Download, GalleryHorizontalEnd, GitBranch, ImageUp, Images, MonitorCog, PanelLeftClose, PanelLeftOpen, RefreshCw, Search, Settings, X } from 'lucide-react';

import { ExtrasPage } from '../views/extras/ExtrasPage';
import { GalleryPage } from '../views/gallery/GalleryPage';
import { HomePage } from '../views/home/HomePage';
import { ModelsPage } from '../views/modelCatalog/ModelsPage';
import { SettingsPage } from '../views/settings/SettingsPage';
import { WorkflowPage } from '../views/workflow/WorkflowPage';
import { useModelCatalogQuery } from '../api/queries';
import { nexusApi } from '../api/nexusClient';
import { useUiStore } from '../stores/uiStore';
import { useGenerationStore } from '../stores/generationStore';
import type { CivitaiModelItem } from '../api/types';

const shellPresets = ['SD', 'SDXL', 'Flux', 'Qwen', 'ZImageTurbo', 'Lumina', 'Wan', 'LTX', 'Krea2', 'Music3', 'Anima', 'Ideogram4'];

function shellPresetLabel(preset: string) {
  if (preset === 'SD') return 'SD 1.5';
  if (preset === 'Wan') return 'WAN 2.2';
  if (preset === 'LTX') return 'LTX 2.3';
  if (preset === 'Ideogram4') return 'IDEOGRAM 4';
  if (preset === 'Krea2') return 'KREA 2';
  if (preset === 'Music3') return 'MUSIC 3';
  return preset === 'ZImageTurbo' ? 'Z-IMG' : preset.toUpperCase();
}

function civitaiBaseModel(preset: string) {
  const values: Record<string, string> = {
    SD: 'SD 1.5',
    SDXL: 'SDXL 1.0',
    Flux: 'Flux.1 D',
    Qwen: 'Qwen Image',
    ZImageTurbo: 'ZImageTurbo',
    Wan: 'WAN 2.2',
    LTX: 'LTXV 2.3',
  };
  return values[preset] || preset;
}

function isVideoPreview(url: string | undefined) {
  return /\.(mp4|webm|mov)(\?|$)/i.test(String(url || ''));
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
    ideogram4: ['ideogram'],
    ideogram: ['ideogram'],
    krea2: ['krea2', 'krea-2', 'krea'],
    'krea-2': ['krea2', 'krea-2', 'krea'],
    music3: ['minimax_music3', 'minimax-music3', 'music3', 'music-3'],
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
  const [civitaiOpen, setCivitaiOpen] = useState(false);
  const [civitaiQuery, setCivitaiQuery] = useState('');
  const [civitaiResults, setCivitaiResults] = useState<CivitaiModelItem[]>([]);
  const [civitaiStatus, setCivitaiStatus] = useState('');
  const checkpoints = checkpointOptions(catalog.data, generation.preset);

  async function runCivitaiLookup() {
    const query = civitaiQuery.trim();
    setCivitaiStatus('Searching Civitai...');
    try {
      if (/^https?:\/\//i.test(query)) {
        const resolved = await nexusApi.civitaiResolve({ url: query, preset: generation.preset, target_kind: 'auto' });
        setCivitaiResults([
          {
            id: String(resolved.model_id || resolved.version_id || query),
            name: String(resolved.model_name || 'Resolved Civitai asset'),
            type: String(resolved.model_type || resolved.target_kind || 'Model'),
            creator: String(resolved.creator || ''),
            preview: String(resolved.preview || ''),
            versions: [{ name: String(resolved.version_name || ''), base_model: String(resolved.base_model || ''), url: String(resolved.url || query), file_name: String(resolved.file_name || '') }],
          },
        ]);
        setCivitaiStatus('URL resolved. Download wiring is available in the backend.');
        return;
      }
      const result = await nexusApi.civitaiSearch({
        query,
        types: '',
        base_model: civitaiBaseModel(generation.preset),
        sort: 'Most Downloaded',
        period: 'AllTime',
        nsfw: true,
        limit: 20,
      });
      setCivitaiResults(result.items || []);
      setCivitaiStatus(`${result.items?.length ?? 0} result(s)`);
    } catch (error) {
      setCivitaiStatus(error instanceof Error ? error.message : 'Civitai lookup failed.');
    }
  }

  return (
    <>
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
          <button className="flat-button compact-action" type="button" onClick={() => setCivitaiOpen(true)}>
            <Download size={13} />
            Civitai
          </button>
        </div>
      </header>
      {civitaiOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Nexus Civitai downloader">
          <div className="modal-panel civitai-modal">
            <header className="modal-header">
              <strong>Nexus Civitai Downloader</strong>
              <button className="icon-button" type="button" onClick={() => setCivitaiOpen(false)} title="Close Civitai downloader">
                <X size={15} />
              </button>
            </header>
            <div className="modal-search-row">
              <Search size={15} />
              <input value={civitaiQuery} onChange={(event) => setCivitaiQuery(event.currentTarget.value)} placeholder="Search terms or paste a Civitai URL..." />
              <button className="primary-button" type="button" onClick={runCivitaiLookup}>Resolve URL</button>
            </div>
            <div className="modal-filter-row">
              {['All Types', 'Checkpoints', 'LoRAs', 'Embeddings'].map((label) => (
                <button className="mini-button" type="button" key={label}>{label}</button>
              ))}
              <span>Base: {civitaiBaseModel(generation.preset)}</span>
              <span>Order: Most Downloaded</span>
            </div>
            <div className="civitai-card-grid">
              {(civitaiResults.length ? civitaiResults : [
                { name: 'Realistic Vision V6.0 B1', type: 'Checkpoint', versions: [{ base_model: 'SD 1.5' }] },
                { name: 'DreamShaper', type: 'Checkpoint', versions: [{ base_model: 'SD 1.5' }] },
                { name: 'Juggernaut XL', type: 'Checkpoint', versions: [{ base_model: 'SDXL' }] },
                { name: 'LTX-2.3 Motion Master', type: 'LoRA', versions: [{ base_model: 'LTX 2.3' }] },
                { name: 'Cyberpunk Aesthetic', type: 'LoRA', versions: [{ base_model: 'Anima' }] },
              ]).map((item) => (
                <article className="civitai-card" key={`${item.id || item.name}`}>
                  {item.preview && !isVideoPreview(item.preview) ? <img src={item.preview} alt={item.name} loading="lazy" decoding="async" /> : <div><span>{item.versions?.[0]?.base_model || item.type}</span></div>}
                  <strong>{item.name}</strong>
                  <small>{item.creator ? `by ${item.creator}` : item.type} {item.versions?.[0]?.file_name ? `/ ${item.versions[0].file_name}` : ''}</small>
                </article>
              ))}
            </div>
            <footer className="modal-footer">
              <span>{civitaiStatus || 'Search uses the local backend Civitai bridge.'}</span>
              <Link className="flat-button" to="/models" onClick={() => setCivitaiOpen(false)}>Open Local Models</Link>
            </footer>
          </div>
        </div>
      )}
    </>
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
