import { create } from 'zustand';

type ExtrasMode = 'image' | 'video' | 'remove_bg';
type RemoveBgPreset = 'image' | 'video';
type ScaleMode = '2x' | '4x' | 'custom';
type EncoderMode = 'mp4_h264' | 'webm_vp9' | 'mov_prores_4444_alpha' | 'mov_prores_422' | 'image_sequence_png' | 'image_sequence_png_alpha';
type ExportFormat = 'png' | 'webp' | 'jpg';
type RemoteSource = {
  url: string;
  title: string;
  mediaType: 'image' | 'video' | 'image_sequence';
};

interface ExtrasState {
  mode: ExtrasMode;
  files: File[];
  remoteSource: RemoteSource | null;
  sourceFps: number;
  targetFps: number;
  preserveAlpha: boolean;
  imageScale: ScaleMode;
  customWidth: number;
  customHeight: number;
  videoUpscaleEnabled: boolean;
  videoInterpolateEnabled: boolean;
  videoDenoiseEnabled: boolean;
  videoDenoiseStrength: number;
  videoScale: '2x' | '4x';
  encoder: EncoderMode;
  removeBgPreset: RemoveBgPreset;
  removeBgThreshold: number;
  removeBgOutput: 'rgba' | 'mask' | 'both';
  imageUpscaler: string;
  videoUpscaleModel: string;
  interpolationModel: string;
  removeBgModel: string;
  exportFormat: ExportFormat;
  tileSize: number;
  setMode: (mode: ExtrasMode) => void;
  setFiles: (files: File[]) => void;
  setRemoteSource: (source: RemoteSource | null) => void;
  setSourceFps: (fps: number) => void;
  setTargetFps: (fps: number) => void;
  setPreserveAlpha: (value: boolean) => void;
  setImageScale: (scale: ScaleMode) => void;
  setCustomSize: (width: number, height: number) => void;
  setVideoUpscaleEnabled: (value: boolean) => void;
  setVideoInterpolateEnabled: (value: boolean) => void;
  setVideoDenoiseEnabled: (value: boolean) => void;
  setVideoDenoiseStrength: (value: number) => void;
  setVideoScale: (scale: '2x' | '4x') => void;
  setEncoder: (encoder: EncoderMode) => void;
  setRemoveBgPreset: (preset: RemoveBgPreset) => void;
  setRemoveBgThreshold: (value: number) => void;
  setRemoveBgOutput: (value: 'rgba' | 'mask' | 'both') => void;
  setImageUpscaler: (model: string) => void;
  setVideoUpscaleModel: (model: string) => void;
  setInterpolationModel: (model: string) => void;
  setRemoveBgModel: (model: string) => void;
  setExportFormat: (format: ExportFormat) => void;
  setTileSize: (tile: number) => void;
  clearSource: () => void;
}

export const useExtrasStore = create<ExtrasState>((set) => ({
  mode: 'image',
  files: [],
  remoteSource: null,
  sourceFps: 24,
  targetFps: 60,
  preserveAlpha: true,
  imageScale: '2x',
  customWidth: 1024,
  customHeight: 1024,
  videoUpscaleEnabled: false,
  videoInterpolateEnabled: true,
  videoDenoiseEnabled: false,
  videoDenoiseStrength: 0.2,
  videoScale: '2x',
  encoder: 'mp4_h264',
  removeBgPreset: 'image',
  removeBgThreshold: 0.45,
  removeBgOutput: 'rgba',
  imageUpscaler: 'Lanczos',
  videoUpscaleModel: 'Lanczos',
  interpolationModel: 'frame_interpolation/rife_v4.26.safetensors',
  removeBgModel: 'BiRefNet',
  exportFormat: 'png',
  tileSize: 512,
  setMode: (mode) => set({ mode }),
  setFiles: (files) => set({ files, remoteSource: null }),
  setRemoteSource: (remoteSource) => set({ remoteSource, files: [] }),
  setSourceFps: (sourceFps) => set({ sourceFps }),
  setTargetFps: (targetFps) => set({ targetFps }),
  setPreserveAlpha: (preserveAlpha) => set({ preserveAlpha }),
  setImageScale: (imageScale) => set({ imageScale }),
  setCustomSize: (customWidth, customHeight) => set({ customWidth, customHeight }),
  setVideoUpscaleEnabled: (videoUpscaleEnabled) => set({ videoUpscaleEnabled }),
  setVideoInterpolateEnabled: (videoInterpolateEnabled) => set({ videoInterpolateEnabled }),
  setVideoDenoiseEnabled: (videoDenoiseEnabled) => set({ videoDenoiseEnabled }),
  setVideoDenoiseStrength: (videoDenoiseStrength) => set({ videoDenoiseStrength }),
  setVideoScale: (videoScale) => set({ videoScale }),
  setEncoder: (encoder) => set({ encoder }),
  setRemoveBgPreset: (removeBgPreset) => set({ removeBgPreset }),
  setRemoveBgThreshold: (removeBgThreshold) => set({ removeBgThreshold }),
  setRemoveBgOutput: (removeBgOutput) => set({ removeBgOutput }),
  setImageUpscaler: (imageUpscaler) => set({ imageUpscaler }),
  setVideoUpscaleModel: (videoUpscaleModel) => set({ videoUpscaleModel }),
  setInterpolationModel: (interpolationModel) => set({ interpolationModel }),
  setRemoveBgModel: (removeBgModel) => set({ removeBgModel }),
  setExportFormat: (exportFormat) => set({ exportFormat }),
  setTileSize: (tileSize) => set({ tileSize }),
  clearSource: () => set({ files: [], remoteSource: null }),
}));
