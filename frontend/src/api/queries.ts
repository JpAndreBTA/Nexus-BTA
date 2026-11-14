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
export function useGalleryIndexQuery() {
  return useInfiniteQuery({
    queryKey: ['gallery-index'],
    initialPageParam: null as string | null,
    queryFn: ({ pageParam }) => nexusApi.galleryPage(pageParam),
    getNextPageParam: (page) => page.next_cursor || undefined,
    staleTime: 10_000,
  });
}
