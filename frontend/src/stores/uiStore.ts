import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type ShellDensity = 'compact' | 'comfortable';
export type NexusTheme = 'graphite' | 'midnight' | 'obsidian';
export type NexusAccent = 'red' | 'blue' | 'violet' | 'emerald' | 'amber';

export interface UiState {
  density: ShellDensity;
  theme: NexusTheme;
  accent: NexusAccent;
  reducedMotion: boolean;
  controlsWidth: number;
  galleryWidth: number;
  sidebarCollapsed: boolean;
  studioControlsCollapsed: boolean;
  studioGalleryOpen: boolean;
  studioGalleryExpanded: boolean;
  setDensity: (density: ShellDensity) => void;
  setTheme: (theme: NexusTheme) => void;
  setAccent: (accent: NexusAccent) => void;
  setReducedMotion: (value: boolean) => void;
  setControlsWidth: (width: number) => void;
  setGalleryWidth: (width: number) => void;
  resetAppearance: () => void;
  toggleSidebar: () => void;
  toggleStudioControls: () => void;
  toggleStudioGallery: () => void;
  toggleStudioGalleryExpanded: () => void;
}
const appearanceDefaults = {
  density: 'compact' as ShellDensity,
  theme: 'graphite' as NexusTheme,
  accent: 'red' as NexusAccent,
  reducedMotion: false,
  controlsWidth: 430,
  galleryWidth: 310,
};
export const useUiStore = create<UiState>()(
  persist((set) => ({
    ...appearanceDefaults,
    sidebarCollapsed: false,
    studioControlsCollapsed: false,
    studioGalleryOpen: true,
    studioGalleryExpanded: false,
    setDensity: (density) => set({ density }),
    setTheme: (theme) => set({ theme }),
    setAccent: (accent) => set({ accent }),
    setReducedMotion: (reducedMotion) => set({ reducedMotion }),
    setControlsWidth: (value) => set({ controlsWidth: Math.round(Math.max(320, Math.min(600, Number.isFinite(value) ? value : 430))) }),
    setGalleryWidth: (value) => set({ galleryWidth: Math.round(Math.max(220, Math.min(480, Number.isFinite(value) ? value : 310))) }),
    resetAppearance: () => set(appearanceDefaults),
    toggleSidebar: () => set((state) => ({ sidebarCollapsed: !state.sidebarCollapsed })),
    toggleStudioControls: () => set((state) => ({ studioControlsCollapsed: !state.studioControlsCollapsed })),
    toggleStudioGallery: () => set((state) => ({ studioGalleryOpen: !state.studioGalleryOpen })),
    toggleStudioGalleryExpanded: () => set((state) => ({ studioGalleryExpanded: !state.studioGalleryExpanded })),
  }), {
    name: 'nexus-ui-state',
    // Preserves existing layout choices and persists no media or browser objects.
    partialize: (state) => ({
      density: state.density, theme: state.theme, accent: state.accent, reducedMotion: state.reducedMotion,
      controlsWidth: state.controlsWidth, galleryWidth: state.galleryWidth,
      sidebarCollapsed: state.sidebarCollapsed, studioControlsCollapsed: state.studioControlsCollapsed,
      studioGalleryOpen: state.studioGalleryOpen, studioGalleryExpanded: state.studioGalleryExpanded,
    }),
  }),
);
