import { useMemo, useState } from 'react';
import { ChevronLeft, ChevronRight, RefreshCw, Search } from 'lucide-react';

import type { GalleryItem } from '../../api/types';
import { useGalleryQuery } from '../../api/queries';

function mediaUrl(value: string) {
  return value.startsWith('/') ? value : `/${value.replace(/^\/+/, '')}`;
}

function isVideo(item: GalleryItem) {
  return item.media_type === 'video' || /\.(mp4|mov|webm|mkv|avi)$/i.test(item.filename);
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
  const gallery = useGalleryQuery();
  const [query, setQuery] = useState('');
  const [sortOrder, setSortOrder] = useState('newest');
  const [pageIndex, setPageIndex] = useState(0);
  const [selected, setSelected] = useState<GalleryItem | null>(null);

  const items = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    const source = gallery.data ?? [];
    const filtered = normalized ? source.filter((item) => `${item.filename} ${item.folder} ${item.model} ${item.preset} ${item.activity}`.toLowerCase().includes(normalized)) : source;
    return [...filtered].sort((a, b) => {
      if (sortOrder === 'oldest') return a.modified - b.modified;
      if (sortOrder === 'name') return a.filename.localeCompare(b.filename);
      if (sortOrder === 'type') return String(a.media_type).localeCompare(String(b.media_type));
      return b.modified - a.modified;
    });
  }, [gallery.data, query, sortOrder]);

  const pageSize = 36;
  const totalPages = Math.max(1, Math.ceil(items.length / pageSize));
  const currentPage = Math.min(pageIndex, totalPages - 1);
  const pagedItems = items.slice(currentPage * pageSize, currentPage * pageSize + pageSize);

  const active = selected ?? pagedItems[0] ?? items[0] ?? null;

  return (
    <section className="page gallery-layout">
      <header className="page-controls">
        <span className="route-sentinel" data-route="gallery" aria-label="Gallery workspace" />
        <button className="flat-button" type="button" onClick={() => gallery.refetch()}>
          <RefreshCw size={14} />
          Refresh
        </button>
      </header>

      <div className="gallery-toolbar">
        <Search size={16} />
        <input value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Search outputs, folders, models..." />
        <select value={sortOrder} onChange={(event) => setSortOrder(event.currentTarget.value)}>
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
          {pagedItems.map((item) => (
            <button key={`${item.relative_path}:${item.modified}`} className={active?.relative_path === item.relative_path ? 'gallery-card active' : 'gallery-card'} type="button" onClick={() => setSelected(item)}>
              {isVideo(item) ? <video src={mediaUrl(item.thumb || item.image)} muted playsInline /> : <img src={mediaUrl(item.thumb || item.image)} alt={item.filename} loading="lazy" />}
              <span title={item.filename}>{item.filename}</span>
            </button>
          ))}
        </div>

        <aside className="surface gallery-preview">
          {active ? (
            <>
              <div className="gallery-preview-media">
                {isVideo(active) ? <video src={mediaUrl(active.image)} controls playsInline /> : <img src={mediaUrl(active.image)} alt={active.filename} />}
              </div>
              <div className="gallery-meta">
                <h2>{active.filename}</h2>
                <p>{active.folder || 'output'}</p>
                <a className="primary-button gallery-send" href={extrasHref(active)}>
                  Send to Extras
                </a>
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
