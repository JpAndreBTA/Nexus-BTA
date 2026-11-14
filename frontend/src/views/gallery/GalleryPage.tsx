import { lazy, Suspense, useEffect, useMemo, useState } from 'react';
import { Box, ChevronLeft, ChevronRight, Music, RefreshCw, Search } from 'lucide-react';

import type { GalleryItem } from '../../api/types';
import { useGalleryIndexQuery } from '../../api/queries';
import { GalleryVideoTile } from './GalleryVideoTile';

const GalleryGlbPreview = lazy(() => import('./GalleryGlbPreview').then((module) => ({ default: module.GalleryGlbPreview })));

function mediaUrl(value: string) {
  return value.startsWith('/') ? value : `/${value.replace(/^\/+/, '')}`;
}

function isVideo(item: GalleryItem) {
  return item.media_type === 'video' || /\.(mp4|mov|webm|mkv|avi)$/i.test(item.filename);
}

function isAudio(item: GalleryItem) {
  return item.media_type === 'audio' || /\.(mp3|wav|flac|opus|ogg|m4a)$/i.test(item.filename);
}

function is3D(item: GalleryItem) {
  return item.media_type === '3d' || /\.(glb|gltf|obj|fbx|stl|ply|usdz|spz|ksplat)$/i.test(item.filename);
}

function extrasHref(item: GalleryItem) {
  const base = window.location.pathname.startsWith('/app') ? '/app' : '';
  const params = new URLSearchParams({
    source: item.image,
    media: isVideo(item) ? 'video' : 'image',
    title: item.filename,
  });
  return `${base}/extras?${params.toString()}`;
}

export function GalleryPage() {
  const [query, setQuery] = useState('');
  const [effectiveQuery, setEffectiveQuery] = useState('');
  const [mediaType, setMediaType] = useState<'all' | 'image' | 'video' | 'audio' | '3d'>('all');
  const [sortOrder, setSortOrder] = useState<'newest' | 'oldest' | 'name' | 'type'>('newest');
  const [refreshId, setRefreshId] = useState(0);
  const gallery = useGalleryIndexQuery({ query: effectiveQuery, media: mediaType, sort: sortOrder, refreshId });
  const loadedItems = useMemo(() => gallery.data?.pages.flatMap((page) => page.items) ?? [], [gallery.data?.pages]);
  const indexedTotal = gallery.data?.pages[0]?.total ?? loadedItems.length;
  const [pageIndex, setPageIndex] = useState(0);
  const [selected, setSelected] = useState<GalleryItem | null>(null);

  // Filtering/sorting occur against the entire SQLite index, not just loaded pages.
  const items = loadedItems;
  useEffect(() => {
    const timer = window.setTimeout(() => setEffectiveQuery(query.trim()), 250);
    return () => window.clearTimeout(timer);
  }, [query]);
  useEffect(() => {
    setPageIndex(0);
    setSelected(null);
  }, [effectiveQuery, mediaType, sortOrder, refreshId]);

  const pageSize = 36;
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const currentPage = Math.min(pageIndex, totalPages - 1);
  const pagedItems = items.slice(currentPage * pageSize, currentPage * pageSize + pageSize);

  const active = (selected ? items.find((item) => item.relative_path === selected.relative_path) : undefined) ?? pagedItems[0] ?? items[0] ?? null;

  return (
    <section className="page gallery-layout">
      <header className="page-controls">
        <span className="route-sentinel" data-route="gallery" aria-label="Gallery workspace" />
        <button className="flat-button" type="button" onClick={() => setRefreshId((id) => id + 1)}>
          <RefreshCw size={14} />
          Refresh
        </button>
      </header>

      <div className="gallery-toolbar">
        <Search size={16} />
        <input value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Search all output filenames and folders..." aria-label="Search all gallery outputs" maxLength={120} />
        <span className="gallery-index-count" title="Search, type and sorting apply to the full SQLite index; loaded items render incrementally.">
          {loadedItems.length} / {indexedTotal} loaded
        </span>
        <select aria-label="Filter gallery media type" value={mediaType}
          onChange={(event) => setMediaType(event.currentTarget.value as typeof mediaType)}>
          <option value="all">All media</option>
          <option value="image">Images</option>
          <option value="video">Videos</option>
          <option value="audio">Audio</option>
          <option value="3d">3D models</option>
        </select>
        <select aria-label="Sort gallery" value={sortOrder}
          onChange={(event) => setSortOrder(event.currentTarget.value as typeof sortOrder)}>
          <option value="newest">Newest</option>
          <option value="oldest">Oldest</option>
          <option value="name">Name</option>
          <option value="type">Type</option>
        </select>
        <button className="mini-button" type="button" onClick={() => setPageIndex((value) => Math.max(0, value - 1))} disabled={currentPage === 0}>
          <ChevronLeft size={14} />
        </button>
        <span>{currentPage + 1} / {totalPages}</span>
        <button className="mini-button" type="button" onClick={() => setPageIndex((value) => Math.min(totalPages - 1, value + 1))} disabled={currentPage >= totalPages - 1}>
          <ChevronRight size={14} />
        </button>
      </div>

      <div className="gallery-shell">
        <div className="gallery-grid">
          {gallery.isError && <div className="studio-inline-status error" role="alert">
            {gallery.error instanceof Error ? gallery.error.message : 'Gallery index unavailable.'}
          </div>}
          {pagedItems.map((item) => (
            <button key={`${item.relative_path}:${item.modified}`} className={active?.relative_path === item.relative_path ? 'gallery-card active' : 'gallery-card'} type="button" onClick={() => setSelected(item)}>
              {isVideo(item) ? (
                <GalleryVideoTile thumbnail={item.thumb || ''} filename={item.filename} />
              ) : isAudio(item) ?
                <div className="gallery-media-placeholder"><Music size={26} /><small>Audio</small></div> : is3D(item) ?
                <div className="gallery-media-placeholder"><Box size={26} /><small>3D asset</small></div> :
                <img src={mediaUrl(item.thumb || item.image)} alt={item.filename} loading="lazy" decoding="async" />}
              <span title={item.filename}>{item.filename}</span>
            </button>
          ))}
          {gallery.hasNextPage && (
            <button className="mini-button gallery-load-more" type="button"
              disabled={gallery.isFetchingNextPage}
              onClick={() => { void gallery.fetchNextPage(); }}>
              {gallery.isFetchingNextPage ? 'Loading more outputs…' : `Load more (${loadedItems.length} / ${indexedTotal})`}
            </button>
          )}
        </div>

        <aside className="surface gallery-preview">
          {active ? (
            <>
              <div className="gallery-preview-media">
                {isVideo(active) ? <video key={active.relative_path} src={mediaUrl(active.image)} controls playsInline preload="metadata" /> : isAudio(active) ?
                  <audio key={active.relative_path} src={mediaUrl(active.image)} controls preload="none" /> : is3D(active) ?
                  /\.glb$/i.test(active.filename)
                    ? <Suspense fallback={<div className="preview-empty"><Box size={30} /><span>Loading experimental GLB viewer…</span></div>}>
                        <GalleryGlbPreview key={active.relative_path} src={mediaUrl(active.image)} filename={active.filename} />
                      </Suspense>
                    : <div className="gallery-media-placeholder large"><Box size={42} /><span>Geometry preview currently supports GLB only.</span>
                        <a className="mini-button" href={mediaUrl(active.image)} download={active.filename}>Download 3D asset</a></div>
                  :
                  <img src={mediaUrl(active.image)} alt={active.filename} decoding="async" />}
              </div>
              <div className="gallery-meta">
                <h2>{active.filename}</h2>
                <p>{active.folder || 'output'}</p>
                {!is3D(active) && !isAudio(active) && <a className="primary-button gallery-send" href={extrasHref(active)}>
                  Send to Extras
                </a>}
                <dl>
                  <div>
                    <dt>Preset</dt>
                    <dd>{active.preset || '-'}</dd>
                  </div>
                  <div>
                    <dt>Model</dt>
                    <dd>{active.model || '-'}</dd>
                  </div>
                  <div>
                    <dt>Seed</dt>
                    <dd>{active.seed || '-'}</dd>
                  </div>
                  <div>
                    <dt>Size</dt>
                    <dd>{active.width && active.height ? `${active.width}x${active.height}` : '-'}</dd>
                  </div>
                </dl>
              </div>
            </>
          ) : (
            <div className="preview-empty">
              <Search size={34} />
              <p>No outputs found.</p>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
