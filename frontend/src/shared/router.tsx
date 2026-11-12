import { lazy, Suspense, useEffect, useRef, useState } from 'react';
import { Link, Outlet, createRootRoute, createRoute, createRouter } from '@tanstack/react-router';
import { Activity, Download, GalleryHorizontalEnd, GitBranch, ImageUp, Images, MonitorCog, PanelLeftClose, PanelLeftOpen, RefreshCw, Search, Settings, X } from 'lucide-react';

const ExtrasPage = lazy(() => import('../views/extras/ExtrasPage').then((module) => ({ default: module.ExtrasPage })));
const GalleryPage = lazy(() => import('../views/gallery/GalleryPage').then((module) => ({ default: module.GalleryPage })));
const HomePage = lazy(() => import('../views/home/HomePage').then((module) => ({ default: module.HomePage })));
const ModelsPage = lazy(() => import('../views/modelCatalog/ModelsPage').then((module) => ({ default: module.ModelsPage })));
const SettingsPage = lazy(() => import('../views/settings/SettingsPage').then((module) => ({ default: module.SettingsPage })));
const WorkflowPage = lazy(() => import('../views/workflow/WorkflowPage').then((module) => ({ default: module.WorkflowPage })));
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
  const civitaiDialogRef = useRef<HTMLDivElement | null>(null);
  const [civitaiFilter, setCivitaiFilter] = useState('All Types');
  const [civitaiDownloadJobId, setCivitaiDownloadJobId] = useState('');
  const [civitaiDownloadStatus, setCivitaiDownloadStatus] = useState('');
  const [civitaiDownloading, setCivitaiDownloading] = useState(false);
  const [civitaiQuery, setCivitaiQuery] = useState('');
  const [civitaiResults, setCivitaiResults] = useState<CivitaiModelItem[]>([]);
  const [civitaiStatus, setCivitaiStatus] = useState('');
  const checkpoints = checkpointOptions(catalog.data, generation.preset);

  useEffect(() => {
    if (!civitaiDownloadJobId) return;
    let disposed = false;
    const poll = async () => {
      try {
        const job = await nexusApi.civitaiDownloadJob(civitaiDownloadJobId);
        if (disposed) return;
        setCivitaiDownloadStatus(`${job.message || job.status} (${Math.round(job.progress || 0)}%)`);
        if (['completed', 'failed', 'cancelled'].includes(job.status)) {
          setCivitaiDownloading(false);
          setCivitaiDownloadJobId('');
        }
      } catch (error) {
        if (disposed) return;
        setCivitaiDownloading(false);
        setCivitaiDownloadJobId('');
        setCivitaiDownloadStatus(error instanceof Error ? error.message : 'Download status unavailable.');
      }
    };
    void poll();
    const timer = window.setInterval(() => { void poll(); }, 1500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [civitaiDownloadJobId]);

  useEffect(() => {
    if (!civitaiOpen) return;
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const dialog = civitaiDialogRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog?.querySelector<HTMLInputElement>('input')?.focus();

    function handleDialogKeydown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault();
        setCivitaiOpen(false);
        return;
      }
      if (event.key !== 'Tab' || !dialog) return;
      const tabbable = Array.from(dialog.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )).filter((element) => element.getClientRects().length > 0);
      if (!tabbable.length) {
        event.preventDefault();
        return;
      }
      const first = tabbable[0];
      const last = tabbable[tabbable.length - 1];
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener('keydown', handleDialogKeydown);
    return () => {
      document.removeEventListener('keydown', handleDialogKeydown);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [civitaiOpen]);

  async function downloadCivitaiModel(url: string) {
    if (!url || civitaiDownloading) return;
    setCivitaiDownloading(true);
    setCivitaiDownloadStatus('Starting download…');
    try {
      const job = await nexusApi.startCivitaiDownload(url, generation.preset);
      setCivitaiDownloadJobId(job.job_id);
    } catch (error) {
      setCivitaiDownloading(false);
      setCivitaiDownloadStatus(error instanceof Error ? error.message : 'Download could not start.');
    }
  }

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
        types: ({ 'All Types': '', Checkpoints: 'Checkpoint', LoRAs: 'LORA', Embeddings: 'TextualInversion' } as Record<string, string>)[civitaiFilter] || '',
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
          <div className="modal-panel civitai-modal" ref={civitaiDialogRef}>
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
                <button className={civitaiFilter === label ? 'mini-button active' : 'mini-button'} type="button" key={label} aria-pressed={civitaiFilter === label} onClick={() => setCivitaiFilter(label)}>{label}</button>
              ))}
              <span>Base: {civitaiBaseModel(generation.preset)}</span>
              <span>Order: Most Downloaded</span>
            </div>
            <div className="civitai-card-grid">
              {!civitaiResults.length && <p className="muted" role="status">No models loaded. Enter a search or a Civitai URL to show verified results.</p>}
              {civitaiResults.map((item) => (
                <article className="civitai-card" key={`${item.id || item.name}`}>
                  {item.preview && !isVideoPreview(item.preview) ? <img src={item.preview} alt={item.name} loading="lazy" decoding="async" /> : <div><span>{item.versions?.[0]?.base_model || item.type}</span></div>}
                  <strong>{item.name}</strong>
                  <small>{item.creator ? `by ${item.creator}` : item.type} {item.versions?.[0]?.file_name ? `/ ${item.versions[0].file_name}` : ''}</small>
                  <button type="button" className="mini-button" disabled={!item.versions?.[0]?.url || civitaiDownloading}
                    onClick={() => { void downloadCivitaiModel(item.versions?.[0]?.url || ''); }}>
                    <Download size={13} /> Download
                  </button>
                </article>
              ))}
            </div>
            <footer className="modal-footer">
              <span role="status" aria-live="polite">{civitaiDownloadStatus || civitaiStatus || 'Search uses the local backend Civitai bridge.'}</span>
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
