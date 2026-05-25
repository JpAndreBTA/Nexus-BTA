import { create } from 'zustand';

export interface ActiveLora {
  name: string;
  relative_name: string;
  relative_path?: string;
  strength: number;
  strength_model: number;
  strength_clip: number;
}

interface LoraState {
  activeLoras: ActiveLora[];
  addLora: (lora: ActiveLora) => void;
  removeLora: (relativeName: string) => void;
  updateStrength: (relativeName: string, strength: number) => void;
  clearLoras: () => void;
}

export const useLoraStore = create<LoraState>((set) => ({
  activeLoras: [],
  addLora: (lora) =>
    set((state) => ({
      activeLoras: [lora, ...state.activeLoras.filter((item) => item.relative_name !== lora.relative_name)],
    })),
  removeLora: (relativeName) => set((state) => ({ activeLoras: state.activeLoras.filter((item) => item.relative_name !== relativeName) })),
  updateStrength: (relativeName, strength) =>
    set((state) => ({
      activeLoras: state.activeLoras.map((item) => (item.relative_name === relativeName ? { ...item, strength, strength_model: strength, strength_clip: strength } : item)),
    })),
  clearLoras: () => set({ activeLoras: [] }),
}));
