import type { ExtrasPlan, ModelCatalog } from '../../api/types';

import type { useExtrasStore } from '../../stores/extrasStore';

type ExtrasState = ReturnType<typeof useExtrasStore.getState>;

const imageExtensions = /\.(png|jpe?g|webp|bmp|gif)$/i;
const videoExtensions = /\.(mp4|mov|webm|mkv|avi)$/i;

export function isImageFile(file: File) {
  return file.type.startsWith('image/') || imageExtensions.test(file.name);
}

export function isVideoFile(file: File) {
  return file.type.startsWith('video/') || videoExtensions.test(file.name);
}

export function sourceTypeForFiles(files: File[]) {
  const hasVideo = files.some(isVideoFile);
  const imageCount = files.filter(isImageFile).length;
  if (hasVideo) return 'video';
  if (imageCount > 1) return 'image_sequence';
  return 'image';
}

export function sourceTypeForState(state: ExtrasState) {
  return state.remoteSource?.mediaType || sourceTypeForFiles(state.files);
}

export function catalogNames(catalog: ModelCatalog | undefined, categories: string[], fallback: string[] = [], filter?: (name: string) => boolean) {
  const names = categories
    .flatMap((category) => catalog?.categories?.[category] ?? [])
    .map((asset) => asset.relative_path || asset.name)
    .filter((name) => name && (!filter || filter(name)));
  return Array.from(new Set([...names, ...fallback]));
}

function boundedNumber(value: number, min: number, max: number, fallback: number) {
  const normalized = Number.isFinite(value) ? value : fallback;
  return Math.max(min, Math.min(max, normalized));
}

export function buildExtrasPlan(state: ExtrasState, catalog: ModelCatalog | undefined): ExtrasPlan {
  const sourceType = sourceTypeForState(state);
  const nvidiaUpscalers = ['nvidia_pid', 'nvidia_rtx'];
  const upscalers = catalogNames(catalog, ['upscale_models'], ['Lanczos', ...nvidiaUpscalers]);
  const videoUpscalers = catalogNames(catalog, ['upscale_models', 'latent_upscale_models'], ['Lanczos', ...nvidiaUpscalers]);
  const interpolators = catalogNames(catalog, ['frame_interpolation'], ['frame_interpolation/rife_v4.26.safetensors']);
  const removeBgModels = catalogNames(catalog, ['background_removal', 'RMBG'], ['BiRefNet']);

  if (state.mode === 'remove_bg') {
    const sourceMediaType = state.removeBgPreset === 'video' || sourceType === 'video' || sourceType === 'image_sequence' ? 'video' : 'image';
    const encoder = sourceMediaType === 'video' ? state.encoder : 'png';
    return {
      mediaType: sourceMediaType,
      mode: 'remove_bg',
      source_type: sourceType,
      source_url: state.remoteSource?.url || '',
      template: sourceMediaType === 'video' ? 'REMOVE_BG_VIDEO_2026' : 'REMOVE_BG_IMAGE_2026',
      preserve_alpha: true,
      encoder,
      pack_metadata: true,
      remove_background: {
        enabled: true,
        model: state.removeBgModel || removeBgModels[0] || 'BiRefNet',
        output: state.removeBgOutput,
        edge_mode: sourceMediaType === 'video' ? 'soft' : 'alpha_matte',
        threshold: boundedNumber(state.removeBgThreshold, 0, 1, 0.45),
        video_frame_batch: sourceMediaType === 'video',
      },
    };
  }

  if (state.mode === 'image') {
    const imageUpscaler = state.imageUpscaler || upscalers[0] || 'Lanczos';
    const imageUpscaleEngine = nvidiaUpscalers.includes(imageUpscaler) ? imageUpscaler : 'standard';
    const customResolution =
      state.imageScale === 'custom'
        ? {
            width: boundedNumber(state.customWidth, 64, 16384, 1024),
            height: boundedNumber(state.customHeight, 64, 16384, 1024),
          }
        : null;
    return {
      mediaType: 'image',
      mode: 'image',
      source_type: sourceType,
      source_url: state.remoteSource?.url || '',
      upscaler: imageUpscaler,
      scale: state.imageScale,
      custom_resolution: customResolution,
      upscale: {
        enabled: true,
        engine: imageUpscaleEngine,
        model: imageUpscaler,
        scale: state.imageScale,
        custom_resolution: customResolution,
      },
      preserve_alpha: state.preserveAlpha,
      export_format: state.exportFormat,
      pack_metadata: true,
    };
  }

  return {
    mediaType: sourceType === 'image_sequence' ? 'image_sequence' : 'video',
    mode: 'video',
    source_type: sourceType,
    source_url: state.remoteSource?.url || '',
    source_fps: sourceType === 'image_sequence' ? boundedNumber(state.sourceFps, 1, 240, 24) : 0,
    preserve_alpha: state.preserveAlpha,
    encoder: state.encoder,
    encode_enabled: true,
    pack_metadata: true,
    interpolate: {
      enabled: state.videoInterpolateEnabled,
      model: state.videoInterpolateEnabled ? state.interpolationModel || interpolators[0] || 'frame_interpolation/rife_v4.26.safetensors' : 'Off',
      fps: boundedNumber(state.targetFps, 1, 240, 60),
      speed: '1x',
    },
    upscale: {
      enabled: state.videoUpscaleEnabled,
      model: state.videoUpscaleEnabled ? state.videoUpscaleModel || videoUpscalers[0] || 'Lanczos' : 'Off',
      engine: state.videoUpscaleEnabled && nvidiaUpscalers.includes(state.videoUpscaleModel) ? state.videoUpscaleModel : 'standard',
      scale: state.videoScale,
      tile: boundedNumber(state.tileSize, 128, 2048, 512),
    },
    denoise: {
      enabled: state.videoDenoiseEnabled,
      strength: boundedNumber(state.videoDenoiseStrength, 0, 1, 0.2),
    },
  };
}

export function extrasPlanLabel(plan: ExtrasPlan) {
  if (plan.mode === 'remove_bg') {
    return `remove bg / ${plan.remove_background?.model || 'BiRefNet'} / ${plan.remove_background?.output || 'rgba'}`;
  }
  if (plan.mode === 'video') {
    const bits = [];
    if (plan.interpolate?.enabled) bits.push(`interp ${plan.interpolate.fps} FPS`);
    if (plan.upscale?.enabled) bits.push(`${plan.upscale.engine || 'standard'} ${plan.upscale.scale}`);
    bits.push(plan.encoder || 'mp4_h264');
    return bits.join(' / ');
  }
  return `${plan.upscaler || 'Lanczos'} / ${plan.scale || '2x'}${plan.preserve_alpha ? ' / alpha' : ''}`;
}
