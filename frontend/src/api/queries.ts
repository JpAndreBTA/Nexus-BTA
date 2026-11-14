import { useInfiniteQuery, useQuery } from '@tanstack/react-query';

import { nexusApi } from './nexusClient';

export function useHealthQuery() {
  return useQuery({
    queryKey: ['health'],
    queryFn: nexusApi.health,
    refetchInterval: 5_000,
  });
}

export function useModelCatalogQuery() {
  return useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      await nexusApi.refreshModelTree();
      return nexusApi.models();
    },
    staleTime: 60_000,
  });
}

export function useGalleryQuery() {
  return useQuery({
    queryKey: ['gallery'],
    queryFn: nexusApi.gallery,
    staleTime: 10_000,
  });
}

export function useLorasQuery() {
  return useQuery({
    queryKey: ['loras'],
    queryFn: nexusApi.loras,
    staleTime: 60_000,
  });
}

export function useWorkflowsQuery() {
  return useQuery({
    queryKey: ['workflows'],
    queryFn: nexusApi.workflows,
    staleTime: 60_000,
  });
}

export function useWorkflowAnalysisQuery(workflowId: string) {
  return useQuery({
    queryKey: ['workflow-analysis', workflowId],
    queryFn: () => nexusApi.workflowAnalysis(workflowId),
    enabled: !!workflowId,
    staleTime: 30_000,
  });
}

/** Cursor-backed gallery for React: the legacy array route stays for Studio and /ui. */
export interface GalleryIndexFilters {
  query?: string;
  media?: 'all' | 'image' | 'video' | 'audio' | '3d';
  sort?: 'newest' | 'oldest' | 'name' | 'type';
  refreshId?: number;
}

export function useGalleryIndexQuery({
  query = '', media = 'all', sort = 'newest', refreshId = 0,
}: GalleryIndexFilters = {}) {
  return useInfiniteQuery({
    queryKey: ['gallery-index', query, media, sort, refreshId],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam, signal }) =>
      nexusApi.galleryPage(pageParam, 48, refreshId > 0 && !pageParam, signal, { query, media, sort }),
    getNextPageParam: (page) => page.next_cursor || undefined,
    staleTime: 10_000,
  });
}

/**
 * Inspect already running ComfyUI; never requests startup/downloads.
 * Only real backend workflow classes are checked (not synthetic editor nodes).
 */
export function useComfyCapabilitiesQuery(classes: string[], enabled: boolean) {
  const key = classes.join(',');
  return useQuery({
    queryKey: ['comfy-capabilities', key],
    queryFn: ({ signal }) => nexusApi.comfyCapabilities(classes, signal),
    enabled: enabled && classes.length > 0,
    staleTime: 30_000,
    retry: 1,
  });
}

/**
 * Backend checks the exact registered classes from the workflow's source JSON.
 * The cached state is keyed by workflow ID; there is no Comfy auto-start.
 */
export function useWorkflowCapabilitiesQuery(workflowId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['comfy-workflow-capabilities', workflowId],
    queryFn: ({ signal }) => nexusApi.workflowCapabilities(workflowId, signal),
    enabled: enabled && !!workflowId,
    staleTime: 30_000,
    retry: 1,
  });
}

/** Saved workflow vs node object_info inputs; no workflow is run or patched. */
export function useWorkflowInputReportQuery(workflowId: string, enabled: boolean) {
  return useQuery({
    queryKey: ['comfy-workflow-inputs', workflowId],
    queryFn: ({ signal }) => nexusApi.workflowInputs(workflowId, signal),
    enabled: enabled && Boolean(workflowId),
    staleTime: 30_000,
    retry: 1,
  });
}
