import { create } from 'zustand';
import { persist } from 'zustand/middleware';

type ShellDensity = 'compact' | 'comfortable';

interface UiState {
  density: ShellDensity;
  sidebarCollapsed: boolean;
  studioControlsCollapsed: boolean;
  studioGalleryOpen: boolean;
  studioGalleryExpanded: boolean;
  setDensity: (density: ShellDensity) => void;
  toggleSidebar: () => void;
  toggleStudioControls: () => void;
  toggleStudioGallery: () => void;
  toggleStudioGalleryExpanded: () => void;
}

export const useUiStore = create<UiState>()(
  persist(
    (set) => ({
      density: 'compact',
      sidebarCollapsed: false,
      studioControlsCollapsed: false,
      studioGalleryOpen: true,
      studioGalleryExpanded: false,
      setDensity: (density) => set({ density }),
      toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
      toggleStudioControls: () => set((state) => ({ studioControlsCollapsed: !state.studioControlsCollapsed })),
      toggleStudioGallery: () => set((state) => ({ studioGalleryOpen: !state.studioGalleryOpen })),
      toggleStudioGalleryExpanded: () => set((state) => ({ studioGalleryExpanded: !state.studioGalleryExpanded })),
    }),
    { name: 'nexus-ui-state' },
  ),
);
