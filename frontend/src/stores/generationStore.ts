import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface ReferenceImage {
  dataUrl: string;
  name: string;
}

export interface GenerationState {
  activity: 'txt2img' | 'img2img';
  img2imgMode: 'image' | 'inpaint';
  preset: string;
  workflowId: string;
  workflowName: string;
  modelPath: string;
  modelName: string;
  prompt: string;
  negativePrompt: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  seed: number;
  denoise: number;
  resizeMode: string;
  maskBlur: number;
  maskContent: string;
  brushSize: number;
  referenceImage: string | null;
  referenceImageName: string;
  extraReferenceImages: ReferenceImage[];
  inpaintMaskImage: string | null;
  controlNetEnabled: boolean;
  controlNetType: string;
  controlNetModel: string;
  controlNetModelName: string;
  controlNetImage: string | null;
  controlNetImageName: string;
  controlNetStrength: number;
  controlNetStart: number;
  controlNetEnd: number;
  controlNetBalance: string;
  videoFrames: number;
  videoFps: number;
  videoSeconds: number;
  videoMotionStrength: number;
  videoActiveAudio: boolean;
  vaeOverrideEnabled: boolean;
  textEncoderOverrideEnabled: boolean;
  audioVaeEnabled: boolean;
  latentUpscaleEnabled: boolean;
  distilledLoraEnabled: boolean;
  lightningLoraEnabled: boolean;
  textEncoder: string;
  videoVae: string;
  audioVae: string;
  latentUpscale: string;
  latentUpscaleRefine: boolean;
  decodeTilesX: number;
  decodeTilesY: number;
  decodeOverlap: number;
  directorEnabled: boolean;
  directorDuration: number;
  directorGuideStrength: number;
  directorLocalPrompt: string;
  directorLocalNegative: string;
  directorResizeMethod: string;
  directorDivisibleBy: number;
  directorImgCompression: number;
  directorUseCustomAudio: boolean;
  setActivity: (activity: 'txt2img' | 'img2img') => void;
  setImg2ImgMode: (mode: 'image' | 'inpaint') => void;
  setPreset: (preset: string) => void;
  setWorkflow: (workflowId: string, workflowName: string) => void;
  setModel: (path: string, name: string) => void;
  setPrompt: (prompt: string) => void;
  setNegativePrompt: (negativePrompt: string) => void;
  setSize: (width: number, height: number) => void;
  setSteps: (steps: number) => void;
  setCfg: (cfg: number) => void;
  setSampler: (sampler: string) => void;
  setScheduler: (scheduler: string) => void;
  setSeed: (seed: number) => void;
  setDenoise: (denoise: number) => void;
  setResizeMode: (resizeMode: string) => void;
  setMaskBlur: (maskBlur: number) => void;
  setMaskContent: (maskContent: string) => void;
  setBrushSize: (brushSize: number) => void;
  setReferenceImage: (dataUrl: string | null, name?: string) => void;
  addExtraReferenceImages: (images: ReferenceImage[]) => void;
  removeExtraReferenceImage: (index: number) => void;
  clearExtraReferenceImages: () => void;
  setInpaintMaskImage: (dataUrl: string | null) => void;
  setControlNetEnabled: (enabled: boolean) => void;
  setControlNetType: (type: string) => void;
  setControlNetModel: (path: string, name: string) => void;
  setControlNetImage: (dataUrl: string | null, name?: string) => void;
  setControlNetStrength: (strength: number) => void;
  setControlNetRange: (start: number, end: number) => void;
  setControlNetBalance: (balance: string) => void;
  setVideoTiming: (frames: number, fps: number, seconds: number) => void;
  setVideoMotionStrength: (strength: number) => void;
  setVideoActiveAudio: (enabled: boolean) => void;
  setVaeOverrideEnabled: (enabled: boolean) => void;
  setTextEncoderOverrideEnabled: (enabled: boolean) => void;
  setAudioVaeEnabled: (enabled: boolean) => void;
  setLatentUpscaleEnabled: (enabled: boolean) => void;
  setDistilledLoraEnabled: (enabled: boolean) => void;
  setLightningLoraEnabled: (enabled: boolean) => void;
  setTextEncoder: (textEncoder: string) => void;
  setVideoVae: (videoVae: string) => void;
  setAudioVae: (audioVae: string) => void;
  setLatentUpscale: (latentUpscale: string) => void;
  setLatentUpscaleRefine: (enabled: boolean) => void;
  setDecodeTiles: (tilesX: number, tilesY: number, overlap: number) => void;
  setDirectorEnabled: (enabled: boolean) => void;
  setDirectorTiming: (duration: number, guideStrength: number) => void;
  setDirectorPrompts: (prompt: string, negative: string) => void;
  setDirectorResize: (method: string, divisibleBy: number, imgCompression: number) => void;
  setDirectorUseCustomAudio: (enabled: boolean) => void;
}

export const useGenerationStore = create<GenerationState>()(
  persist(
    (set) => ({
  activity: 'txt2img',
  img2imgMode: 'image',
  preset: 'SD',
  workflowId: '',
  workflowName: '',
  modelPath: '',
  modelName: '',
  prompt: '',
  negativePrompt: '',
  width: 1024,
  height: 576,
  steps: 25,
  cfg: 7,
  sampler: 'euler_ancestral',
  scheduler: 'karras',
  seed: -1,
  denoise: 0.75,
  resizeMode: 'Just Resize',
  maskBlur: 8,
  maskContent: 'Original',
  brushSize: 42,
  referenceImage: null,
  referenceImageName: '',
  extraReferenceImages: [],
  inpaintMaskImage: null,
  controlNetEnabled: false,
  controlNetType: 'canny',
  controlNetModel: 'Automatic',
  controlNetModelName: 'Automatic',
  controlNetImage: null,
  controlNetImageName: '',
  controlNetStrength: 0.75,
  controlNetStart: 0,
  controlNetEnd: 1,
  controlNetBalance: 'Balanced',
  videoFrames: 81,
  videoFps: 16,
  videoSeconds: 5,
  videoMotionStrength: 0.85,
  videoActiveAudio: false,
  vaeOverrideEnabled: false,
  textEncoderOverrideEnabled: false,
  audioVaeEnabled: true,
  latentUpscaleEnabled: true,
  distilledLoraEnabled: true,
  lightningLoraEnabled: true,
  textEncoder: 'Automatic',
  videoVae: 'Automatic',
  audioVae: 'Automatic',
  latentUpscale: 'Automatic',
  latentUpscaleRefine: true,
  decodeTilesX: 2,
  decodeTilesY: 2,
  decodeOverlap: 6,
  directorEnabled: false,
  directorDuration: 12,
  directorGuideStrength: 1,
  directorLocalPrompt: '',
  directorLocalNegative: '',
  directorResizeMethod: 'maintain aspect ratio',
  directorDivisibleBy: 32,
  directorImgCompression: 18,
  directorUseCustomAudio: false,
  setActivity: (activity) => set({ activity }),
  setImg2ImgMode: (img2imgMode) => set({ img2imgMode }),
  setPreset: (preset) =>
    set((state) => {
      const lower = preset.toLowerCase();
      if (lower === 'wan') {
        return {
          ...state,
          preset,
          width: 512,
          height: 512,
          steps: 4,
          cfg: 1,
          sampler: 'euler',
          scheduler: 'simple',
          videoFrames: 81,
          videoFps: 16,
          videoSeconds: 5,
          videoMotionStrength: 0.85,
          latentUpscaleEnabled: false,
        };
      }
      if (lower === 'ltx') {
        return {
          ...state,
          preset,
          width: 512,
          height: 512,
          steps: 8,
          cfg: 1,
          sampler: 'euler_ancestral_cfg_pp',
          scheduler: 'quadratic',
          videoFrames: 121,
          videoFps: 24,
          videoSeconds: 5,
          videoMotionStrength: 0.85,
          audioVaeEnabled: true,
          latentUpscaleEnabled: true,
          distilledLoraEnabled: true,
          latentUpscaleRefine: true,
          decodeTilesX: 2,
          decodeTilesY: 2,
          decodeOverlap: 6,
        };
      }
      return { preset };
    }),
  setWorkflow: (workflowId, workflowName) => set({ workflowId, workflowName }),
  setModel: (modelPath, modelName) => set({ modelPath, modelName }),
  setPrompt: (prompt) => set({ prompt }),
  setNegativePrompt: (negativePrompt) => set({ negativePrompt }),
  setSize: (width, height) => set({ width, height }),
  setSteps: (steps) => set({ steps }),
  setCfg: (cfg) => set({ cfg }),
  setSampler: (sampler) => set({ sampler }),
  setScheduler: (scheduler) => set({ scheduler }),
  setSeed: (seed) => set({ seed }),
  setDenoise: (denoise) => set({ denoise }),
  setResizeMode: (resizeMode) => set({ resizeMode }),
  setMaskBlur: (maskBlur) => set({ maskBlur }),
  setMaskContent: (maskContent) => set({ maskContent }),
  setBrushSize: (brushSize) => set({ brushSize }),
  setReferenceImage: (referenceImage, referenceImageName = '') => set({ referenceImage, referenceImageName, inpaintMaskImage: null }),
  addExtraReferenceImages: (images) => set((state) => ({ extraReferenceImages: [...state.extraReferenceImages, ...images].slice(0, 6) })),
  removeExtraReferenceImage: (index) => set((state) => ({ extraReferenceImages: state.extraReferenceImages.filter((_, itemIndex) => itemIndex !== index) })),
  clearExtraReferenceImages: () => set({ extraReferenceImages: [] }),
  setInpaintMaskImage: (inpaintMaskImage) => set({ inpaintMaskImage }),
  setControlNetEnabled: (controlNetEnabled) => set({ controlNetEnabled }),
  setControlNetType: (controlNetType) => set({ controlNetType }),
  setControlNetModel: (controlNetModel, controlNetModelName) => set({ controlNetModel, controlNetModelName }),
  setControlNetImage: (controlNetImage, controlNetImageName = '') => set({ controlNetImage, controlNetImageName }),
  setControlNetStrength: (controlNetStrength) => set({ controlNetStrength }),
  setControlNetRange: (controlNetStart, controlNetEnd) => set({ controlNetStart, controlNetEnd }),
  setControlNetBalance: (controlNetBalance) => set({ controlNetBalance }),
  setVideoTiming: (videoFrames, videoFps, videoSeconds) => set({ videoFrames, videoFps, videoSeconds }),
  setVideoMotionStrength: (videoMotionStrength) => set({ videoMotionStrength }),
  setVideoActiveAudio: (videoActiveAudio) => set({ videoActiveAudio }),
  setVaeOverrideEnabled: (vaeOverrideEnabled) => set({ vaeOverrideEnabled }),
  setTextEncoderOverrideEnabled: (textEncoderOverrideEnabled) => set({ textEncoderOverrideEnabled }),
  setAudioVaeEnabled: (audioVaeEnabled) => set({ audioVaeEnabled }),
  setLatentUpscaleEnabled: (latentUpscaleEnabled) => set({ latentUpscaleEnabled }),
  setDistilledLoraEnabled: (distilledLoraEnabled) => set({ distilledLoraEnabled }),
  setLightningLoraEnabled: (lightningLoraEnabled) => set({ lightningLoraEnabled }),
  setTextEncoder: (textEncoder) => set({ textEncoder }),
  setVideoVae: (videoVae) => set({ videoVae }),
  setAudioVae: (audioVae) => set({ audioVae }),
  setLatentUpscale: (latentUpscale) => set({ latentUpscale }),
  setLatentUpscaleRefine: (latentUpscaleRefine) => set({ latentUpscaleRefine }),
  setDecodeTiles: (decodeTilesX, decodeTilesY, decodeOverlap) => set({ decodeTilesX, decodeTilesY, decodeOverlap }),
  setDirectorEnabled: (directorEnabled) => set({ directorEnabled }),
  setDirectorTiming: (directorDuration, directorGuideStrength) => set({ directorDuration, directorGuideStrength }),
  setDirectorPrompts: (directorLocalPrompt, directorLocalNegative) => set({ directorLocalPrompt, directorLocalNegative }),
  setDirectorResize: (directorResizeMethod, directorDivisibleBy, directorImgCompression) => set({ directorResizeMethod, directorDivisibleBy, directorImgCompression }),
  setDirectorUseCustomAudio: (directorUseCustomAudio) => set({ directorUseCustomAudio }),
    }),
    {
      name: 'nexus-generation-state',
      partialize: (state) => ({
        activity: state.activity,
        img2imgMode: state.img2imgMode,
        preset: state.preset,
        workflowId: state.workflowId,
        workflowName: state.workflowName,
        modelPath: state.modelPath,
        modelName: state.modelName,
        prompt: state.prompt,
        negativePrompt: state.negativePrompt,
        width: state.width,
        height: state.height,
        steps: state.steps,
        cfg: state.cfg,
        sampler: state.sampler,
        scheduler: state.scheduler,
        seed: state.seed,
        denoise: state.denoise,
        resizeMode: state.resizeMode,
        maskBlur: state.maskBlur,
        maskContent: state.maskContent,
        brushSize: state.brushSize,
        controlNetEnabled: state.controlNetEnabled,
        controlNetType: state.controlNetType,
        controlNetModel: state.controlNetModel,
        controlNetModelName: state.controlNetModelName,
        controlNetStrength: state.controlNetStrength,
        controlNetStart: state.controlNetStart,
        controlNetEnd: state.controlNetEnd,
        controlNetBalance: state.controlNetBalance,
        videoFrames: state.videoFrames,
        videoFps: state.videoFps,
        videoSeconds: state.videoSeconds,
        videoMotionStrength: state.videoMotionStrength,
        videoActiveAudio: state.videoActiveAudio,
        vaeOverrideEnabled: state.vaeOverrideEnabled,
        textEncoderOverrideEnabled: state.textEncoderOverrideEnabled,
        audioVaeEnabled: state.audioVaeEnabled,
        latentUpscaleEnabled: state.latentUpscaleEnabled,
        distilledLoraEnabled: state.distilledLoraEnabled,
        lightningLoraEnabled: state.lightningLoraEnabled,
        textEncoder: state.textEncoder,
        videoVae: state.videoVae,
        audioVae: state.audioVae,
        latentUpscale: state.latentUpscale,
        latentUpscaleRefine: state.latentUpscaleRefine,
        decodeTilesX: state.decodeTilesX,
        decodeTilesY: state.decodeTilesY,
        decodeOverlap: state.decodeOverlap,
        directorEnabled: state.directorEnabled,
        directorDuration: state.directorDuration,
        directorGuideStrength: state.directorGuideStrength,
        directorLocalPrompt: state.directorLocalPrompt,
        directorLocalNegative: state.directorLocalNegative,
        directorResizeMethod: state.directorResizeMethod,
        directorDivisibleBy: state.directorDivisibleBy,
        directorImgCompression: state.directorImgCompression,
        directorUseCustomAudio: state.directorUseCustomAudio,
      }),
    },
  ),
);
