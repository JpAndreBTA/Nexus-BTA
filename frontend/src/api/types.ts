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
    scale: string;
    tile?: number;
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
