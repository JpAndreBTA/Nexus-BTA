import { useMemo } from 'react';

import { useModelCatalogQuery } from '../../api/queries';

export function ModelsPage() {
  const catalog = useModelCatalogQuery();
  const categories = useMemo(() => Object.entries(catalog.data?.categories ?? {}), [catalog.data]);

  return (
    <section className="page">
      <header className="page-header">
        <div>
          <p className="eyebrow">Catalog</p>
          <h1>Models</h1>
        </div>
        <span className="status-pill">{catalog.data?.total_files ?? 0} files</span>
      </header>

      <div className="model-category-grid">
        {categories.map(([name, assets]) => (
          <article className="surface model-category" key={name}>
            <h3>{name}</h3>
            <p className="muted">{assets.length} detected</p>
          </article>
        ))}
      </div>
    </section>
  );
}
