import type { ExtrasJob, ExtrasPlan, GalleryItem, GenerateRequest, GenerationJob, HealthResponse, LoraAsset, ModelCatalog } from './types';

const API_BASE = (import.meta.env.NEXUS_API_URL || '/api').replace(/\/$/, '');

async function parseResponse<T>(response: Response): Promise<T> {
  if (!response.ok) {
    const text = await response.text();
    try {
      const parsed = JSON.parse(text) as { detail?: unknown };
      throw new Error(typeof parsed.detail === 'string' ? parsed.detail : JSON.stringify(parsed.detail ?? parsed));
    } catch (error) {
      if (error instanceof SyntaxError) {
        throw new Error(text || response.statusText);
      }
      throw error;
    }
  }
  return response.json() as Promise<T>;
}

export async function nexusFetch<T>(path: string, init?: RequestInit): Promise<T> {
  return parseResponse<T>(await fetch(`${API_BASE}${path}`, init));
}

export const nexusApi = {
  health: () => nexusFetch<HealthResponse>('/health'),
  refreshModelTree: () => nexusFetch<{ ok?: boolean }>('/model-tree', { method: 'POST' }),
  models: () => nexusFetch<ModelCatalog>('/models'),
  loras: () => nexusFetch<LoraAsset[]>('/loras'),
  startExtras: async (mode: string, plan: ExtrasPlan, files: File[]) => {
    const form = new FormData();
    form.append('mode', mode);
    form.append('plan', JSON.stringify(plan));
    files.forEach((file) => form.append('files', file, file.name));
    return parseResponse<ExtrasJob>(
      await fetch(`${API_BASE}/extras/start`, {
        method: 'POST',
        body: form,
      }),
    );
  },
  extrasJob: (jobId: string) => nexusFetch<ExtrasJob>(`/extras/${encodeURIComponent(jobId)}`),
  gallery: () => nexusFetch<GalleryItem[]>('/gallery'),
  startGeneration: (payload: GenerateRequest) =>
    nexusFetch<GenerationJob>('/generate/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  generationJob: (jobId: string) => nexusFetch<GenerationJob>(`/generate/${encodeURIComponent(jobId)}`),
  cancelGeneration: (jobId: string) => nexusFetch<GenerationJob>(`/generate/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
};
