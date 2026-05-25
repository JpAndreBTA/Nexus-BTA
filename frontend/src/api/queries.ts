import { useQuery } from '@tanstack/react-query';

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
