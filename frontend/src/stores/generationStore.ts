import { create } from 'zustand';

interface GenerationState {
  preset: string;
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
  setPreset: (preset: string) => void;
  setModel: (path: string, name: string) => void;
  setPrompt: (prompt: string) => void;
  setNegativePrompt: (negativePrompt: string) => void;
  setSize: (width: number, height: number) => void;
  setSteps: (steps: number) => void;
  setCfg: (cfg: number) => void;
  setSampler: (sampler: string) => void;
  setScheduler: (scheduler: string) => void;
  setSeed: (seed: number) => void;
}

export const useGenerationStore = create<GenerationState>((set) => ({
  preset: 'SD',
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
  setPreset: (preset) => set({ preset }),
  setModel: (modelPath, modelName) => set({ modelPath, modelName }),
  setPrompt: (prompt) => set({ prompt }),
  setNegativePrompt: (negativePrompt) => set({ negativePrompt }),
  setSize: (width, height) => set({ width, height }),
  setSteps: (steps) => set({ steps }),
  setCfg: (cfg) => set({ cfg }),
  setSampler: (sampler) => set({ sampler }),
  setScheduler: (scheduler) => set({ scheduler }),
  setSeed: (seed) => set({ seed }),
}));
