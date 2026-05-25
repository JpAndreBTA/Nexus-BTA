import { useMemo, useState } from 'react';
import { RefreshCw, Search } from 'lucide-react';

import type { CatalogAsset } from '../../api/types';
import { useModelCatalogQuery } from '../../api/queries';

function formatBytes(value: number) {
  if (!value) return '-';
  const gb = value / 1024 ** 3;
  if (gb >= 1) return `${gb.toFixed(2)} GB`;
  return `${(value / 1024 ** 2).toFixed(1)} MB`;
}

export function ModelsPage() {
  const catalog = useModelCatalogQuery();
  const [query, setQuery] = useState('');
  const [category, setCategory] = useState('all');
  const [selected, setSelected] = useState<CatalogAsset | null>(null);

  const categories = useMemo(() => Object.entries(catalog.data?.categories ?? {}).filter(([, assets]) => assets.length > 0), [catalog.data]);
  const flatAssets = useMemo(
    () =>
      categories
        .flatMap(([name, assets]) => assets.map((asset) => ({ ...asset, category: asset.category || name })))
        .sort((a, b) => a.relative_path.localeCompare(b.relative_path)),
    [categories],
  );
  const visibleAssets = useMemo(() => {
    const normalized = query.trim().toLowerCase();
    return flatAssets.filter((asset) => {
      const categoryOk = category === 'all' || asset.category === category;
      const haystack = `${asset.name} ${asset.relative_path} ${asset.folder} ${asset.tags?.join(' ')}`.toLowerCase();
      return categoryOk && (!normalized || haystack.includes(normalized));
    });
  }, [category, flatAssets, query]);
  const active = selected ?? visibleAssets[0] ?? null;

  return (
    <section className="page models-layout">
      <header className="page-controls">
        <span className="route-sentinel" data-route="models" aria-label="Models workspace" />
        <button className="flat-button" type="button" onClick={() => catalog.refetch()}>
          <RefreshCw size={14} />
          Refresh
        </button>
      </header>

      <div className="gallery-toolbar">
        <Search size={16} />
        <input value={query} onChange={(event) => setQuery(event.currentTarget.value)} placeholder="Search models, LoRAs, encoders..." />
        <span>{visibleAssets.length} / {catalog.data?.total_files ?? 0}</span>
      </div>

      <div className="models-shell">
        <aside className="surface category-list">
          <button className={category === 'all' ? 'active' : ''} type="button" onClick={() => setCategory('all')}>
            All <span>{flatAssets.length}</span>
          </button>
          {categories.map(([name, assets]) => (
            <button key={name} className={category === name ? 'active' : ''} type="button" onClick={() => setCategory(name)}>
              {name} <span>{assets.length}</span>
            </button>
          ))}
        </aside>

        <main className="model-table">
          {visibleAssets.map((asset) => (
            <button key={asset.relative_path || asset.path} className={active?.relative_path === asset.relative_path ? 'model-row active' : 'model-row'} type="button" onClick={() => setSelected(asset)}>
              <span>{asset.name}</span>
              <em>{asset.category}</em>
              <strong>{formatBytes(asset.size_bytes)}</strong>
            </button>
          ))}
        </main>

        <aside className="surface model-detail">
          {active ? (
            <>
              <h2>{active.name}</h2>
              <dl>
                <div>
                  <dt>Category</dt>
                  <dd>{active.category}</dd>
                </div>
                <div>
                  <dt>Folder</dt>
                  <dd>{active.folder || '-'}</dd>
                </div>
                <div>
                  <dt>Size</dt>
                  <dd>{formatBytes(active.size_bytes)}</dd>
                </div>
                <div>
                  <dt>Source</dt>
                  <dd>{active.source || '-'}</dd>
                </div>
                <div>
                  <dt>Relative</dt>
                  <dd>{active.relative_path}</dd>
                </div>
              </dl>
              <div className="tag-list">
                {(active.tags ?? []).map((tag) => (
                  <span key={tag}>{tag}</span>
                ))}
              </div>
            </>
          ) : (
            <div className="preview-empty">
              <Search size={34} />
              <p>No models found.</p>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}
