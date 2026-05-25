import { create } from 'zustand';

type ShellDensity = 'compact' | 'comfortable';

interface UiState {
  density: ShellDensity;
  sidebarCollapsed: boolean;
  setDensity: (density: ShellDensity) => void;
  toggleSidebar: () => void;
}

export const useUiStore = create<UiState>((set) => ({
  density: 'compact',
  sidebarCollapsed: false,
  setDensity: (density) => set({ density }),
  toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
}));
