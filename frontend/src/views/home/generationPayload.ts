import type { GenerateRequest } from '../../api/types';
import type { GenerationState } from '../../stores/generationStore';
import type { ActiveLora } from '../../stores/loraStore';

/**
 * Pure projection from Studio state to the existing FastAPI generation wire format.
 * Do not mutate source state or remove fields without backend contract tests.
 * Specific preset capability checks stay at the callsite until M3 capability registry.
 */
export interface GenerationPayloadContext {
  generation: GenerationState;
  activeLoras: ActiveLora[];
  allReferenceImages: string[];
  alignedFrames: number;
  directorFrames: number;
  controlNetCompatible: boolean;
  directorMode: boolean;
  ideogram4Mode: boolean;
  music3Mode: boolean;
  videoMode: boolean;
  music3RecommendedProfile?: string;
}

export function buildGenerationPayload({
  generation, activeLoras, allReferenceImages, alignedFrames, directorFrames,
  controlNetCompatible, directorMode, ideogram4Mode, music3Mode, videoMode,
  music3RecommendedProfile,
}: GenerationPayloadContext): GenerateRequest {
  const loraStore = { activeLoras };
  const payload = {

    activity: directorMode ? 'txt2img' : generation.activity,
    workspace: directorMode ? 'director' : 'viewer',
    preset: generation.preset,
    template: directorMode ? 'LTX_DIRECTOR_SUITE' : undefined,
    workflow_id: generation.workflowId || null,
    model_path: generation.modelPath,
    model_name: generation.modelName,
    prompt: generation.prompt,
    negative_prompt: generation.negativePrompt,
    width: generation.width,
    height: generation.height,
    steps: generation.steps,
    cfg: generation.cfg,
    sampler: generation.sampler,
    scheduler: generation.scheduler,
    seed: generation.seed,
    batch_size: 1,
    denoise: generation.activity === 'img2img' ? generation.denoise : 1,
    vae: generation.vaeOverrideEnabled ? generation.videoVae : 'Automatic',
    text_encoder: generation.textEncoderOverrideEnabled ? generation.textEncoder : 'Automatic',
    loras: loraStore.activeLoras,
    distilled_loras: generation.preset.toLowerCase() === 'ltx' && !generation.distilledLoraEnabled ? [{ name: 'None', strength: 0 }] : [],
    music: music3Mode ? {
      caption: generation.musicCaption || generation.prompt,
      mode: generation.musicMode,
      lyrics: generation.musicMode === 'instrumental' ? '[instrumental]' : generation.musicMode === 'lyrics' ? generation.musicLyrics : '',
      negative_prompt: generation.musicNegativePrompt,
      duration_seconds: generation.musicDurationSeconds,
      steps: generation.steps,
      cfg_scale: generation.musicCfgScale,
      top_k: generation.musicTopK,
      tiled_decode: generation.musicTiledDecode,
      format: generation.musicFormat,
    } : {},
    img2img: {
      mode: generation.img2imgMode === 'inpaint' ? 'Inpaint masked area' : 'Image to Image',
      resize_mode: generation.resizeMode,
      denoise: generation.denoise,
      batch_count: 1,
      mask_blur: generation.maskBlur,
      mask_content: generation.maskContent,
      inpaint_engine: generation.inpaintEngine,
      differential_diffusion: generation.inpaintEngine === 'differential',
      differential_strength: generation.differentialStrength,
      lanpaint_thinking_steps: generation.lanpaintThinkingSteps,
      lanpaint_prompt_mode: 'Image First',
      reference_image: generation.referenceImage,
      reference_images: allReferenceImages,
      mask_image: generation.img2imgMode === 'inpaint' ? generation.inpaintMaskImage : null,
    },
    controlnet: {
      enabled: controlNetCompatible && generation.controlNetEnabled,
      type: generation.controlNetType,
      model: generation.controlNetModel || generation.controlNetModelName || 'Automatic',
      image: generation.controlNetImage || generation.referenceImage || null,
      strength: generation.controlNetStrength,
      start_percent: generation.controlNetStart,
      end_percent: generation.controlNetEnd,
      preprocessor: 'Auto',
      low_threshold: 0.4,
      high_threshold: 0.8,
      balance: generation.controlNetBalance,
    },
    video: videoMode || ideogram4Mode
      ? {
          ...(ideogram4Mode
            ? {
                ideogram_regions: generation.promptRegions,
                ideogram_reference_mode: generation.activity === 'img2img' ? 'layout_reference_only' : 'txt2img',
              }
            : {}),
          ...(videoMode
            ? {
          frames: alignedFrames,
          fps: generation.videoFps,
          seconds: generation.videoSeconds,
          duration: generation.videoSeconds,
          motion_adapter: generation.preset.toLowerCase() === 'wan' ? 'WAN T2V / I2V' : 'LTX latent video',
          motion_strength: generation.videoMotionStrength,
          active_audio: generation.videoActiveAudio,
          video_vae: generation.vaeOverrideEnabled ? generation.videoVae : 'Automatic',
          audio_vae: generation.audioVaeEnabled ? generation.audioVae : 'None',
          latent_upscale: generation.latentUpscaleEnabled ? generation.latentUpscale : 'None',
          latent_upscale_refine: generation.latentUpscaleRefine,
          qwen_auto_edit_lora: generation.lightningLoraEnabled,
          decode_tiles_x: generation.decodeTilesX,
          decode_tiles_y: generation.decodeTilesY,
          decode_overlap: generation.decodeOverlap,
          director_timeline: directorMode,
            }
            : {}),
        }
      : {},
    director: directorMode
      ? {
          duration_frames: Math.max(1, directorFrames - 1),
          duration_seconds: generation.directorDuration,
          frame_rate: generation.videoFps,
          local_prompts: generation.directorLocalPrompt || generation.prompt,
          local_negative_prompts: generation.directorLocalNegative || generation.negativePrompt,
          segment_lengths: String(generation.directorDuration),
          epsilon: 0.001,
          guide_strength: generation.directorGuideStrength,
          use_custom_audio: generation.directorUseCustomAudio,
          display_mode: 'seconds',
          custom_width: generation.width,
          custom_height: generation.height,
          resize_method: generation.directorResizeMethod,
          divisible_by: generation.directorDivisibleBy,
          img_compression: generation.directorImgCompression,
          timeline_data: {
            duration: generation.directorDuration,
            fps: generation.videoFps,
            references: allReferenceImages.map((_src, index) => ({ id: `ref-${index + 1}`, start: index === 0 ? 0 : generation.directorDuration, end: generation.directorDuration })),
            audioSegments: [],
          },
          timeline_data_json: JSON.stringify({
            duration: generation.directorDuration,
            fps: generation.videoFps,
            references: allReferenceImages.map((_src, index) => ({ id: `ref-${index + 1}`, start: index === 0 ? 0 : generation.directorDuration, end: generation.directorDuration })),
            audioSegments: [],
          }),
        }
      : {},
    runtime: ideogram4Mode
      ? {
          vram_policy: 'shared',
          attention_backend: 'pytorch',
          disable_xformers: false,
          enable_sage_attention: false,
          enable_flash_attention: false,
          precision: 'auto',
        }
      : music3Mode
        ? { vram_policy: music3RecommendedProfile === 'rtx_5090' ? 'normal' : 'low', attention_backend: 'sage', disable_xformers: false, enable_sage_attention: true, precision: music3RecommendedProfile === 'rtx_5090' ? 'fp16' : 'int8' }
      : {},
  };
  return payload;
}

/**
 * Block invalid dispatch before it reaches FastAPI. No alerts, state writes,
 * or asynchronous effects are allowed in this validation layer.
 */
export interface GenerationValidationContext {
  generation: GenerationState;
  payload: GenerateRequest;
  ideogram4Mode: boolean;
  videoMode: boolean;
  controlNetCompatible: boolean;
  controlNetModelCount: number;
}

export function validateGeneration({
  generation, payload, ideogram4Mode, videoMode, controlNetCompatible,
  controlNetModelCount,
}: GenerationValidationContext): string | null {
  if (!generation.prompt.trim()) return 'Prompt is required.';
  if (!generation.modelPath && !generation.modelName && !ideogram4Mode) {
    return 'Select a model before generating.';
  }
  if (generation.activity === 'img2img' && !generation.referenceImage) {
    return 'Load a reference image for img2img.';
  }
  if (generation.activity === 'img2img' && generation.img2imgMode === 'inpaint' && !generation.inpaintMaskImage) {
    return 'Paint an inpaint mask before generating.';
  }
  if (ideogram4Mode && generation.activity === 'img2img' && generation.img2imgMode === 'inpaint') {
    return 'Ideogram 4 local route does not support true mask inpaint yet. Use Linear Viewer ADD boxes as regional guides.';
  }
  if (videoMode && generation.activity === 'img2img' && !generation.referenceImage) {
    return `${generation.preset} img2video requires a reference image.`;
  }
  if (controlNetCompatible && generation.controlNetEnabled && !payload.controlnet?.image) {
    return 'Load a ControlNet image or use the img2img reference.';
  }
  if (controlNetCompatible && generation.controlNetEnabled && controlNetModelCount === 0 && generation.controlNetModel === 'Automatic') {
    return 'No compatible ControlNet model was detected for this preset and type.';
  }
  return null;
}
