export type MediaKind = 'image' | 'video' | string;

export interface HealthResponse {
  nexus: string;
  comfy_running: boolean;
  comfy_url: string;
  comfy_root_exists: boolean;
  comfy_python_exists: boolean;
  models_dir: string;
  custom_nodes_dir: string;
}

export interface CatalogAsset {
  name: string;
  path: string;
  relative_path: string;
  category: string;
  folder: string;
  extension: string;
  size_bytes: number;
  modified: string;
  tags?: string[];
  source?: string;
  preview?: string;
}

export type LoraAsset = CatalogAsset;

export interface ModelCatalog {
  root: string;
  categories: Record<string, CatalogAsset[]>;
  total_files: number;
}

export interface CivitaiModelItem {
  id?: number | string;
  name: string;
  type: string;
  creator?: string;
  preview?: string;
  tags?: string[];
  stats?: Record<string, unknown>;
  versions?: Array<{
    id?: number | string;
    name?: string;
    base_model?: string;
    url?: string;
    preview?: string;
    file_name?: string;
  }>;
}

export interface CivitaiSearchResponse {
  items: CivitaiModelItem[];
  metadata?: Record<string, unknown>;
}

export interface OutputItem {
  kind: MediaKind;
  filename: string;
  subfolder: string;
  type: string;
  path: string;
  url: string;
}

export interface ExtrasJob {
  job_id: string;
  prompt_id: string | null;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string;
  progress: number;
  message: string;
  outputs: OutputItem[];
  error: string | null;
  created_at: string;
  updated_at: string;
  preset: string;
  workflow_id: string;
  completed_at?: string;
}

export interface ExtrasPlan {
  mediaType: 'image' | 'video' | 'image_sequence' | string;
  mode: 'image' | 'video' | 'remove_bg' | string;
  template?: string;
  source_type?: string;
  source_fps?: number;
  source_url?: string;
  preserve_alpha?: boolean;
  export_format?: string;
  encoder?: string;
  encode_enabled?: boolean;
  pack_metadata?: boolean;
  upscaler?: string;
  scale?: string;
  custom_resolution?: { width: number; height: number } | null;
  upscale?: {
    enabled: boolean;
    model: string;
    engine?: string;
    scale: string;
    tile?: number;
    custom_resolution?: { width: number; height: number } | null;
    runtime_engine?: string;
    fallback_engine?: string;
    fallback_reason?: string;
  };
  interpolate?: {
    enabled: boolean;
    model: string;
    fps: number;
    speed: string;
  };
  remove_background?: {
    enabled: boolean;
    model: string;
    threshold: number;
    output: string;
    edge_mode?: string;
    video_frame_batch?: boolean;
  };
  denoise?: {
    enabled: boolean;
    strength: number;
  };
}

export interface NvidiaExtrasStatus {
  engine: string;
  label: string;
  installed: boolean;
  node_ready: boolean;
  dependency_ready: boolean;
  expected_nodes: string[];
  packages: Record<string, boolean>;
  model_required: boolean;
  models_auto_download: boolean;
  notes?: string;
}

export interface TemplateAssetStatusItem {
  key: string;
  label: string;
  filename: string;
  url?: string;
  kind?: string;
  scope?: string;
  destination?: string;
  path?: string;
  installed: boolean;
  size_bytes_min?: number;
  size_bytes?: number;
}

export interface Ideogram4AssetsStatus {
  template: 'Ideogram4' | string;
  label: string;
  installed: boolean;
  generation_ready: boolean;
  dependencies_installed: boolean;
  assets: TemplateAssetStatusItem[];
  missing_assets: TemplateAssetStatusItem[];
  missing_required_assets: TemplateAssetStatusItem[];
  missing_optional_assets: TemplateAssetStatusItem[];
  missing_custom_node_dependencies?: string[];
  missing_core_nodes?: string[];
  runtime_checked?: boolean;
  estimated_missing_required_bytes?: number;
  estimated_missing_optional_bytes?: number;
  note?: string;
}

export interface GalleryItem {
  title: string;
  filename: string;
  path: string;
  relative_path: string;
  folder: string;
  image: string;
  thumb: string;
  media_type: 'image' | 'video' | string;
  prompt?: string;
  negative?: string;
  model?: string;
  seed?: string;
  steps?: string;
  cfg?: string;
  sampler?: string;
  scheduler?: string;
  preset?: string;
  activity?: string;
  width?: string | number;
  height?: string | number;
  metadata?: Record<string, unknown>;
  modified: number;
}

export interface GenerateRequest {
  activity: string;
  workspace: string;
  preset: string;
  workflow_id?: string | null;
  model_path?: string;
  model_name?: string;
  prompt: string;
  negative_prompt: string;
  width: number;
  height: number;
  steps: number;
  cfg: number;
  sampler: string;
  scheduler: string;
  seed: number;
  batch_size: number;
  denoise: number;
  vae: string;
  text_encoder: string;
  loras: object[];
  distilled_loras: Array<Record<string, unknown>>;
  img2img?: Record<string, unknown>;
  controlnet?: Record<string, unknown>;
  video: Record<string, unknown>;
  director?: Record<string, unknown>;
  runtime?: Record<string, unknown>;
}

export interface Ideogram4PromptJsonRequest {
  prompt: string;
  width: number;
  height: number;
  regions: unknown[];
  provider: 'template' | 'comfy_gemma4' | 'ollama' | 'ideogram_magic' | 'openai_compatible';
  model?: string;
  endpoint?: string;
}

export interface Ideogram4PromptJsonResponse {
  provider: string;
  model: string;
  json_prompt: Record<string, unknown>;
  prompt_text: string;
  used_fallback: boolean;
  message: string;
}

export interface Ideogram4OllamaStatus {
  running: boolean;
  installed: boolean;
  model: string;
  endpoint: string;
  installed_size_bytes?: number;
  estimated_size_bytes?: number;
  message: string;
}

export interface DownloadJob {
  job_id: string;
  kind?: string;
  status: string;
  progress: number;
  message: string;
  error?: string | null;
  bytes_downloaded?: number;
  bytes_total?: number;
  model?: string;
  created_at?: string;
  updated_at?: string;
  completed_at?: string;
}

export interface WorkflowSummary {
  id: string;
  name: string;
  path: string;
  format: 'api' | 'ui' | 'unknown' | string;
  node_count: number;
  class_types: string[];
  tags: string[];
}

export interface WorkflowGraphNode {
  id: string;
  class_type: string;
  title?: string;
  x?: number;
  y?: number;
  width?: number;
  height?: number;
  inputs?: string[];
  outputs?: string[];
  widgets?: Array<{ name?: string; value?: unknown }>;
  bypassed?: boolean;
}

export interface WorkflowGraphLink {
  from_node: string;
  from_slot?: number;
  to_node: string;
  to_slot?: number;
  type?: string;
}

export interface WorkflowVisualGraph {
  nodes?: WorkflowGraphNode[];
  links?: WorkflowGraphLink[];
  groups?: unknown[];
  width?: number;
  height?: number;
}

export interface WorkflowAnalysis {
  workflow: WorkflowSummary;
  missing_nodes: string[];
  available_nodes: number;
  manager_suggestions: Record<string, string[]>;
  dependency_targets: string[];
  dependencies_installed: string[];
  dependency_errors: Record<string, string>;
  visual_graph: WorkflowVisualGraph;
  workflow_settings: Record<string, unknown>;
}

export interface GenerationJob {
  job_id: string;
  prompt_id: string | null;
  status: 'queued' | 'running' | 'completed' | 'failed' | 'cancelled' | string;
  progress: number;
  message: string;
  outputs: OutputItem[];
  error: string | null;
  queue_position?: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  preset?: string;
  workflow_id?: string | null;
}
