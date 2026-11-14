import { lazy, Suspense, useCallback, useState } from 'react';
import { Download, RefreshCw } from 'lucide-react';

import { useModelCatalogQuery } from '../api/queries';
import { useGenerationStore } from '../stores/generationStore';

const CivitaiModal = lazy(() => import('../views/models/CivitaiModal').then(({ CivitaiModal }) => ({ default: CivitaiModal })));

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

export function AppTopbar() {
  const catalog = useModelCatalogQuery();
  const generation = useGenerationStore();
  const [civitaiOpen, setCivitaiOpen] = useState(false);
  const closeCivitai = useCallback(() => setCivitaiOpen(false), []);
  const checkpoints = checkpointOptions(catalog.data, generation.preset);

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
      {civitaiOpen && <Suspense fallback={null}><CivitaiModal preset={generation.preset} onClose={closeCivitai} /></Suspense>}
    </>
  );
}
