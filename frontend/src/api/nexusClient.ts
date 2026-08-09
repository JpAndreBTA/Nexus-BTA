import type { CivitaiSearchResponse, DownloadJob, ExtrasJob, ExtrasPlan, GalleryItem, GenerateRequest, GenerationJob, HealthResponse, Ideogram4AssetsStatus, Ideogram4OllamaStatus, Ideogram4PromptJsonRequest, Ideogram4PromptJsonResponse, Krea2AssetsStatus, LoraAsset, ModelCatalog, NvidiaExtrasStatus, WorkflowAnalysis, WorkflowSummary } from './types';

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
  civitaiSearch: (payload: Record<string, unknown>) =>
    nexusFetch<CivitaiSearchResponse>('/civitai/search', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  civitaiResolve: (payload: Record<string, unknown>) =>
    nexusFetch<Record<string, unknown>>('/civitai/resolve', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
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
  nvidiaExtrasStatus: (engine: string) => nexusFetch<NvidiaExtrasStatus>(`/extras/nvidia/${encodeURIComponent(engine)}/status`),
  ideogram4AssetsStatus: () => nexusFetch<Ideogram4AssetsStatus>('/ideogram4/assets/status'),
  krea2AssetsStatus: () => nexusFetch<Krea2AssetsStatus>('/krea2/assets/status'),
  startKrea2AssetDownload: (payload: Record<string, unknown>) =>
    nexusFetch<GenerationJob>('/krea2/assets/download/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  startIdeogram4AssetDownload: (payload: Record<string, unknown>) =>
    nexusFetch<GenerationJob>('/ideogram4/assets/download/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  ideogram4PromptJson: (payload: Ideogram4PromptJsonRequest) =>
    nexusFetch<Ideogram4PromptJsonResponse>('/ideogram4/prompt-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  ideogram4OllamaStatus: (model: string, endpoint = '') =>
    nexusFetch<Ideogram4OllamaStatus>(`/ideogram4/ollama/status?model=${encodeURIComponent(model)}&endpoint=${encodeURIComponent(endpoint)}`),
  startIdeogram4OllamaPull: (payload: { model: string; endpoint?: string }) =>
    nexusFetch<DownloadJob>('/ideogram4/ollama/pull/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  ideogram4OllamaPullJob: (jobId: string) => nexusFetch<DownloadJob>(`/ideogram4/ollama/pull/${encodeURIComponent(jobId)}`),
  extrasJob: (jobId: string) => nexusFetch<ExtrasJob>(`/extras/${encodeURIComponent(jobId)}`),
  gallery: () => nexusFetch<GalleryItem[]>('/gallery'),
  workflows: () => nexusFetch<WorkflowSummary[]>('/workflows'),
  workflowAnalysis: (workflowId: string) => nexusFetch<WorkflowAnalysis>(`/workflows/${encodeURIComponent(workflowId)}/analysis`),
  loadWorkflow: async (file: File) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return parseResponse<WorkflowAnalysis>(
      await fetch(`${API_BASE}/workflows/load`, {
        method: 'POST',
        body: form,
      }),
    );
  },
  importWorkflow: async (file: File) => {
    const form = new FormData();
    form.append('file', file, file.name);
    return parseResponse<WorkflowAnalysis>(
      await fetch(`${API_BASE}/workflows/import`, {
        method: 'POST',
        body: form,
      }),
    );
  },
  startGeneration: (payload: GenerateRequest) =>
    nexusFetch<GenerationJob>('/generate/start', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }),
  generationJob: (jobId: string) => nexusFetch<GenerationJob>(`/generate/${encodeURIComponent(jobId)}`),
  cancelGeneration: (jobId: string) => nexusFetch<GenerationJob>(`/generate/${encodeURIComponent(jobId)}/cancel`, { method: 'POST' }),
};
