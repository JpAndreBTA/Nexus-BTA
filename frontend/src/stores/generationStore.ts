import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface GenerationState {
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
  setInpaintMaskImage: (dataUrl: string | null) => void;
  setControlNetEnabled: (enabled: boolean) => void;
  setControlNetType: (type: string) => void;
  setControlNetModel: (path: string, name: string) => void;
  setControlNetImage: (dataUrl: string | null, name?: string) => void;
  setControlNetStrength: (strength: number) => void;
  setControlNetRange: (start: number, end: number) => void;
  setControlNetBalance: (balance: string) => void;
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
  setActivity: (activity) => set({ activity }),
  setImg2ImgMode: (img2imgMode) => set({ img2imgMode }),
  setPreset: (preset) => set({ preset }),
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
  setInpaintMaskImage: (inpaintMaskImage) => set({ inpaintMaskImage }),
  setControlNetEnabled: (controlNetEnabled) => set({ controlNetEnabled }),
  setControlNetType: (controlNetType) => set({ controlNetType }),
  setControlNetModel: (controlNetModel, controlNetModelName) => set({ controlNetModel, controlNetModelName }),
  setControlNetImage: (controlNetImage, controlNetImageName = '') => set({ controlNetImage, controlNetImageName }),
  setControlNetStrength: (controlNetStrength) => set({ controlNetStrength }),
  setControlNetRange: (controlNetStart, controlNetEnd) => set({ controlNetStart, controlNetEnd }),
  setControlNetBalance: (controlNetBalance) => set({ controlNetBalance }),
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
      }),
    },
  ),
);
