import type { CatalogAsset } from '../../api/types';
import type { useModelCatalogQuery } from '../../api/queries';

// Centralized discovery and preset compatibility for Studio.
// Keep this mapping aligned with the backend capability registry until the
// M3 capability-driven model selection migration is implemented.
export function modelOptions(catalog: ReturnType<typeof useModelCatalogQuery>['data']) {
  return [...(catalog?.categories.checkpoints ?? []), ...(catalog?.categories.unet ?? []), ...(catalog?.categories.diffusion_models ?? [])].map((asset) => ({
    label: asset.relative_path || asset.name,
    value: asset.relative_path || asset.path,
    name: asset.name,
  }));
}

export function assetSelectOptions(catalog: ReturnType<typeof useModelCatalogQuery>['data'], categories: string[], tokens: string[] = []) {
  const lowered = tokens.map((token) => token.toLowerCase());
  return categories
    .flatMap((category) => ((catalog?.categories[category] ?? []) as CatalogAsset[]))
    .filter((asset) => {
      if (!lowered.length) return true;
      const haystack = `${asset.name} ${asset.folder} ${asset.relative_path} ${asset.tags?.join(' ')}`.toLowerCase();
      return lowered.some((token) => haystack.includes(token));
    })
    .sort((a, b) => (a.relative_path || a.name).localeCompare(b.relative_path || b.name))
    .map((asset) => ({
      label: asset.relative_path || asset.name,
      value: asset.relative_path || asset.path || asset.name,
    }));
}

function modelMatchesPreset(model: { label: string; name: string }, preset: string) {
  const haystack = `${model.label} ${model.name}`.toLowerCase();
  const key = preset.toLowerCase();
  if (['music3', 'music 3', 'minimax_music3', 'minimax-music-3'].includes(key)) return undefined;
  const rules: Record<string, string[]> = {
    sd: ['sd15', 'sd1', '1.5', 'dreamshaper'],
    xl: ['sdxl', 'xl', 'illustrious'],
    sdxl: ['sdxl', 'xl', 'illustrious'],
    flux: ['flux'],
    qwen: ['qwen'],
    ideogram4: ['ideogram'],
    ideogram: ['ideogram'],
    zimageturbo: ['z_image', 'zimage', 'z-image'],
    lumina: ['lumina'],
    wan: ['wan'],
    ltx: ['ltx'],
    anima: ['anima'],
    krea2: ['krea2', 'krea-2', 'krea'],
    music3: ['minimax_music3', 'minimax-music3', 'music3', 'music-3'],
  };
  return (rules[key] || [key]).some((token) => haystack.includes(token));
}

export function primaryModelMatchesPreset(model: { label: string; name: string }, preset: string) {
  const haystack = `${model.label} ${model.name}`.toLowerCase();
  if (['ideogram4', 'ideogram'].includes(preset.toLowerCase())) {
    return modelMatchesPreset(model, preset) && !haystack.includes('unconditional');
  }
  return modelMatchesPreset(model, preset);
}

export function controlNetCompatiblePreset(preset: string) {
  return ['sd', 'sd15', 'xl', 'sdxl', 'ltx', 'flux', 'qwen', 'zimageturbo', 'zimage'].includes(preset.toLowerCase());
}

export function videoPreset(preset: string) {
  return ['wan', 'ltx'].includes(preset.toLowerCase());
}

export function alignVideoFrames(preset: string, frames: number) {
  const step = preset.toLowerCase() === 'ltx' ? 8 : 4;
  const safe = Math.max(1, Math.round(frames || 1));
  return safe % step === 1 ? safe : Math.max(1, Math.round((safe - 1) / step) * step + 1);
}


export function controlNetModelOptions(catalog: ReturnType<typeof useModelCatalogQuery>['data'], preset: string, type: string) {
  const lowerPreset = preset.toLowerCase();
  if (lowerPreset === 'ltx') {
    return ((catalog?.categories.loras ?? []) as CatalogAsset[])
      .filter((model) => {
        const haystack = `${model.name} ${model.folder} ${model.relative_path} ${model.tags?.join(' ')}`.toLowerCase();
        return haystack.includes('ltx') && ['ic-lora', 'ic_lora', 'control', 'cameraman', 'reference'].some((token) => haystack.includes(token));
      })
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((model) => ({
        label: (model.relative_path || model.name).replace(/^loras[\\/]/i, ''),
        value: (model.relative_path || model.path || model.name).replace(/^loras[\\/]/i, '').replaceAll('/', '\\'),
        name: (model.relative_path || model.name).replace(/^loras[\\/]/i, '').replaceAll('/', '\\'),
      }));
  }

  if (['qwen', 'zimageturbo', 'zimage'].includes(lowerPreset)) {
    const categories = [
      ...(((catalog?.categories.model_patches ?? []) as CatalogAsset[])),
      ...(((catalog?.categories.controlnet ?? []) as CatalogAsset[])),
    ];
    const presetTokens = lowerPreset === 'qwen'
      ? ['qwen', 'diffsynth', 'instantx']
      : ['zimage', 'z-image', 'z_image', 'fun', 'controlnet-union'];
    return categories
      .filter((model) => {
        const haystack = `${model.name} ${model.folder} ${model.relative_path} ${model.tags?.join(' ')}`.toLowerCase();
        return presetTokens.some((token) => haystack.includes(token));
      })
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((model) => ({
        label: model.relative_path || model.name,
        value: (model.relative_path || model.path || model.name).replace(/^controlnet[\\/]/i, '').replace(/^model_patches[\\/]/i, '').replaceAll('/', '\\'),
        name: (model.relative_path || model.name).replace(/^controlnet[\\/]/i, '').replace(/^model_patches[\\/]/i, '').replaceAll('/', '\\'),
      }));
  }

  const presetTokens = lowerPreset === 'flux'
    ? ['flux', 'flux.1', 'flux1', 'union']
    : (['xl', 'sdxl'].includes(lowerPreset) ? ['sdxl', 'xl'] : ['sd15', 'sd1', 'v11', '1.5']);
  const typeTokens: Record<string, string[]> = {
    canny: ['canny'],
    depth: ['depth'],
    dwpose: ['dwpose', 'dw pose', 'openpose', 'pose'],
    openpose: ['openpose', 'pose'],
    lineart: ['lineart', 'line'],
    tile: ['tile'],
  };
  const tokens = typeTokens[type] || [type];
  const controlCategories = [
    ...(((catalog?.categories.controlnet ?? []) as CatalogAsset[])),
    ...(((catalog?.categories.model_patches ?? []) as CatalogAsset[])),
  ];
  return controlCategories
    .filter((model) => {
      const haystack = `${model.name} ${model.folder} ${model.relative_path} ${model.tags?.join(' ')}`.toLowerCase();
      const customSource = String(model.source || '').toLowerCase() === 'reference';
      if (lowerPreset === 'flux' && haystack.includes('union')) return customSource || presetTokens.some((token) => haystack.includes(token));
      return (customSource || presetTokens.some((token) => haystack.includes(token))) && tokens.some((token) => haystack.includes(token));
    })
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((model) => ({
      label: model.relative_path || model.name,
      value: (model.relative_path || model.path || model.name).replace(/^controlnet[\\/]/i, '').replace(/^model_patches[\\/]/i, '').replaceAll('/', '\\'),
      name: (model.relative_path || model.name).replace(/^controlnet[\\/]/i, '').replace(/^model_patches[\\/]/i, '').replaceAll('/', '\\'),
    }));
}

