import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Clapperboard, Image as ImageIcon, LoaderCircle, Maximize2, Minimize2, PanelLeftClose, PanelLeftOpen, PanelRightClose, PanelRightOpen, Play, Send, Server, Sparkles, Square, WandSparkles } from 'lucide-react';

import { nexusApi } from '../../api/nexusClient';
import type { CatalogAsset, GenerateRequest, GenerationJob } from '../../api/types';
import { useGalleryQuery, useHealthQuery, useModelCatalogQuery } from '../../api/queries';
import { useLorasQuery } from '../../api/queries';
import { queryClient } from '../../shared/queryClient';
import { useGenerationStore } from '../../stores/generationStore';
import { useLoraStore } from '../../stores/loraStore';
import { useUiStore } from '../../stores/uiStore';
import { InpaintCanvas } from './InpaintCanvas';

function terminalStatus(job: GenerationJob | undefined) {
  return job?.status === 'completed' || job?.status === 'failed' || job?.status === 'cancelled';
}

function outputUrl(url: string | undefined) {
  if (!url) return '';
  return url.startsWith('/') ? url : `/${url.replace(/^\/+/, '')}`;
}

function readFileAsDataUrl(file: File) {
  return new Promise<string>((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result || ''));
    reader.onerror = () => reject(reader.error || new Error('Failed to read image file.'));
    reader.readAsDataURL(file);
  });
}

function modelOptions(catalog: ReturnType<typeof useModelCatalogQuery>['data']) {
  return [...(catalog?.categories.checkpoints ?? []), ...(catalog?.categories.unet ?? []), ...(catalog?.categories.diffusion_models ?? [])].map((asset) => ({
    label: asset.relative_path || asset.name,
    value: asset.relative_path || asset.path,
    name: asset.name,
  }));
}

function modelMatchesPreset(model: { label: string; name: string }, preset: string) {
  const haystack = `${model.label} ${model.name}`.toLowerCase();
  const key = preset.toLowerCase();
  const rules: Record<string, string[]> = {
    sd: ['sd15', 'sd1', '1.5', 'dreamshaper'],
    xl: ['sdxl', 'xl', 'illustrious'],
    sdxl: ['sdxl', 'xl', 'illustrious'],
    flux: ['flux'],
    qwen: ['qwen'],
    zimageturbo: ['z_image', 'zimage', 'z-image'],
    lumina: ['lumina'],
    wan: ['wan'],
    ltx: ['ltx'],
    anima: ['anima'],
  };
  return (rules[key] || [key]).some((token) => haystack.includes(token));
}

function controlNetCompatiblePreset(preset: string) {
  return ['sd', 'sd15', 'xl', 'sdxl', 'ltx'].includes(preset.toLowerCase());
}

function videoPreset(preset: string) {
  return ['wan', 'ltx'].includes(preset.toLowerCase());
}

function alignVideoFrames(preset: string, frames: number) {
  const step = preset.toLowerCase() === 'ltx' ? 8 : 4;
  const safe = Math.max(1, Math.round(frames || 1));
  return safe % step === 1 ? safe : Math.max(1, Math.round((safe - 1) / step) * step + 1);
}

const presetOptions = ['Anima', 'SD', 'XL', 'SDXL', 'Flux', 'Qwen', 'ZImageTurbo', 'Lumina', 'Wan', 'LTX'];

function presetIcon(preset: string) {
  if (['Wan', 'LTX'].includes(preset)) return <Clapperboard size={14} />;
  if (preset === 'Anima') return <Sparkles size={14} />;
  if (preset === 'Qwen' || preset === 'Flux') return <WandSparkles size={14} />;
  return <ImageIcon size={14} />;
}

function controlNetModelOptions(catalog: ReturnType<typeof useModelCatalogQuery>['data'], preset: string, type: string) {
  const lowerPreset = preset.toLowerCase();
  if (lowerPreset === 'ltx') {
    return ((catalog?.categories.loras ?? []) as CatalogAsset[])
      .filter((model) => {
        const haystack = `${model.name} ${model.folder} ${model.relative_path} ${model.tags?.join(' ')}`.toLowerCase();
        return haystack.includes('ltx') && ['ic-lora', 'ic_lora', 'control', 'cameraman', 'reference'].some((token) => haystack.includes(token));
      })
      .sort((a, b) => a.name.localeCompare(b.name))
      .map((model) => ({
        label: (model.relative_path || model.name).replace(/^loras[\\/]/i, ''),
        value: model.relative_path || model.path,
        name: (model.relative_path || model.name).replace(/^loras[\\/]/i, '').replaceAll('/', '\\'),
      }));
  }

  const presetTokens = ['xl', 'sdxl'].includes(lowerPreset) ? ['sdxl', 'xl'] : ['sd15', 'sd1', 'v11', '1.5'];
  const typeTokens: Record<string, string[]> = {
    canny: ['canny'],
    depth: ['depth'],
    openpose: ['openpose', 'pose'],
    lineart: ['lineart', 'line'],
    tile: ['tile'],
  };
  const tokens = typeTokens[type] || [type];
  return ((catalog?.categories.controlnet ?? []) as CatalogAsset[])
    .filter((model) => {
      const haystack = `${model.name} ${model.folder} ${model.relative_path} ${model.tags?.join(' ')}`.toLowerCase();
      return presetTokens.some((token) => haystack.includes(token)) && tokens.some((token) => haystack.includes(token));
    })
    .sort((a, b) => a.name.localeCompare(b.name))
    .map((model) => ({
      label: model.relative_path || model.name,
      value: model.relative_path || model.path,
      name: model.name,
    }));
}

export function HomePage() {
  const health = useHealthQuery();
  const catalog = useModelCatalogQuery();
  const gallery = useGalleryQuery();
  const loras = useLorasQuery();
  const generation = useGenerationStore();
  const loraStore = useLoraStore();
  const ui = useUiStore();
  const [jobId, setJobId] = useState('');
  const [localError, setLocalError] = useState('');
  const [loraSearch, setLoraSearch] = useState('');

  const allModels = useMemo(() => modelOptions(catalog.data), [catalog.data]);
  const models = useMemo(() => {
    const filtered = allModels.filter((model) => modelMatchesPreset(model, generation.preset));
    return filtered.length ? filtered : allModels;
  }, [allModels, generation.preset]);
  const newestGalleryItem = gallery.data?.[0];
  const controlNetCompatible = controlNetCompatiblePreset(generation.preset);
  const videoMode = videoPreset(generation.preset);
  const alignedFrames = alignVideoFrames(generation.preset, generation.videoFrames);
  const controlNetModels = useMemo(() => controlNetModelOptions(catalog.data, generation.preset, generation.controlNetType), [catalog.data, generation.preset, generation.controlNetType]);
  const visibleLoras = useMemo(() => {
    const q = loraSearch.trim().toLowerCase();
    return (loras.data ?? [])
      .filter((lora) => {
        const haystack = `${lora.name} ${lora.relative_path} ${lora.folder} ${lora.tags?.join(' ')}`.toLowerCase();
        return !q || haystack.includes(q);
      })
      .slice(0, 24);
  }, [loras.data, loraSearch]);

  useEffect(() => {
    if ((!generation.modelPath || !models.some((model) => model.value === generation.modelPath)) && models[0]) {
      generation.setModel(models[0].value, models[0].name);
    }
  }, [generation, models]);

  useEffect(() => {
    if (!controlNetCompatible && generation.controlNetEnabled) {
      generation.setControlNetEnabled(false);
    }
    if (generation.controlNetModel !== 'Automatic' && !controlNetModels.some((model) => model.value === generation.controlNetModel)) {
      generation.setControlNetModel('Automatic', 'Automatic');
    }
  }, [controlNetCompatible, controlNetModels, generation]);

  const payload = useMemo<GenerateRequest>(
    () => ({
      activity: generation.activity,
      workspace: 'viewer',
      preset: generation.preset,
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
      vae: 'Automatic',
      text_encoder: 'Automatic',
      loras: loraStore.activeLoras,
      distilled_loras: [],
      img2img: {
        mode: generation.img2imgMode === 'inpaint' ? 'Inpaint masked area' : 'Image to Image',
        resize_mode: generation.resizeMode,
        denoise: generation.denoise,
        batch_count: 1,
        mask_blur: generation.maskBlur,
        mask_content: generation.maskContent,
        reference_image: generation.referenceImage,
        reference_images: generation.referenceImage ? [generation.referenceImage] : [],
        mask_image: generation.img2imgMode === 'inpaint' ? generation.inpaintMaskImage : null,
      },
      controlnet: {
        enabled: controlNetCompatible && generation.controlNetEnabled,
        type: generation.controlNetType,
        model: generation.controlNetModelName || generation.controlNetModel || 'Automatic',
        image: generation.controlNetImage || generation.referenceImage || null,
        strength: generation.controlNetStrength,
        start_percent: generation.controlNetStart,
        end_percent: generation.controlNetEnd,
        preprocessor: 'Auto',
        low_threshold: 0.4,
        high_threshold: 0.8,
        balance: generation.controlNetBalance,
      },
      video: videoMode
        ? {
            frames: alignedFrames,
            fps: generation.videoFps,
            seconds: generation.videoSeconds,
            duration: generation.videoSeconds,
            motion_adapter: generation.preset.toLowerCase() === 'wan' ? 'WAN T2V / I2V' : 'LTX latent video',
            motion_strength: generation.videoMotionStrength,
            active_audio: generation.videoActiveAudio,
            video_vae: generation.videoVae,
            audio_vae: generation.audioVae,
            latent_upscale: generation.latentUpscale,
            latent_upscale_refine: generation.latentUpscaleRefine,
            decode_tiles_x: generation.decodeTilesX,
            decode_tiles_y: generation.decodeTilesY,
            decode_overlap: generation.decodeOverlap,
          }
        : {},
      director: {},
      runtime: {},
    }),
    [alignedFrames, controlNetCompatible, generation, loraStore.activeLoras, videoMode],
  );

  const startMutation = useMutation({
    mutationFn: () => nexusApi.startGeneration(payload),
    onSuccess: (job) => {
      setJobId(job.job_id);
      setLocalError('');
    },
    onError: (error) => setLocalError(error instanceof Error ? error.message : 'Generation failed to start.'),
  });

  const jobQuery = useQuery({
    queryKey: ['generation-job', jobId],
    queryFn: () => nexusApi.generationJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const job = query.state.data as GenerationJob | undefined;
      return job && terminalStatus(job) ? false : 900;
    },
  });

  const cancelMutation = useMutation({
    mutationFn: () => nexusApi.cancelGeneration(jobId),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['generation-job', jobId] }),
  });

  const activeJob = jobQuery.data ?? startMutation.data;
  const processing = startMutation.isPending || (!!activeJob && !terminalStatus(activeJob));
  const generatedOutput = activeJob?.status === 'completed' ? activeJob.outputs?.[0] : null;
  const previewUrl = outputUrl(generatedOutput?.url || newestGalleryItem?.image);
  const previewIsVideo = /\.(mp4|mov|webm|mkv|avi)$/i.test(generatedOutput?.filename || newestGalleryItem?.filename || '');

  async function generate() {
    if (!generation.prompt.trim()) {
      setLocalError('Prompt is required.');
      return;
    }
    if (!generation.modelPath && !generation.modelName) {
      setLocalError('Select a model before generating.');
      return;
    }
    if (generation.activity === 'img2img' && !generation.referenceImage) {
      setLocalError('Load a reference image for img2img.');
      return;
    }
    if (generation.activity === 'img2img' && generation.img2imgMode === 'inpaint' && !generation.inpaintMaskImage) {
      setLocalError('Paint an inpaint mask before generating.');
      return;
    }
    if (videoMode && generation.activity === 'img2img' && !generation.referenceImage) {
      setLocalError(`${generation.preset} img2video requires a reference image.`);
      return;
    }
    if (controlNetCompatible && generation.controlNetEnabled && !payload.controlnet?.image) {
      setLocalError('Load a ControlNet image or use the img2img reference.');
      return;
    }
    if (controlNetCompatible && generation.controlNetEnabled && !controlNetModels.length && generation.controlNetModel === 'Automatic') {
      setLocalError('No compatible ControlNet model was detected for this preset and type.');
      return;
    }
    setLocalError('');
    setJobId('');
    await startMutation.mutateAsync();
  }

  return (
    <section
      className={[
        'page studio-layout',
        ui.studioGalleryOpen ? 'studio-gallery-open' : '',
        ui.studioGalleryExpanded ? 'studio-gallery-expanded' : '',
        ui.studioControlsCollapsed ? 'studio-controls-collapsed' : '',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      <header className="page-header">
        <div>
          <p className="eyebrow">Nexus BTA Web App</p>
          <h1>Studio</h1>
        </div>
        <div className="studio-header-tools">
          <div className="preset-strip" aria-label="Preset quick select">
            {presetOptions.map((preset) => (
              <button className={generation.preset === preset ? 'active' : ''} type="button" key={preset} onClick={() => generation.setPreset(preset)}>
                {presetIcon(preset)}
                <span>{preset}</span>
              </button>
            ))}
          </div>
          <button className="status-pill" type="button" onClick={() => ui.toggleStudioControls()}>
            {ui.studioControlsCollapsed ? <PanelLeftOpen size={14} /> : <PanelLeftClose size={14} />}
            Controls
          </button>
          <button className="status-pill" type="button" onClick={() => ui.toggleStudioGallery()}>
            {ui.studioGalleryOpen ? <PanelRightClose size={14} /> : <PanelRightOpen size={14} />}
            Gallery
          </button>
          <span className={health.data?.nexus === 'ok' ? 'status-pill ok' : 'status-pill'}>
            <Server size={14} />
            {health.data?.nexus === 'ok' ? 'Backend online' : 'Backend check'}
          </span>
        </div>
      </header>

      <div className="studio-columns">
        <aside className="surface tool-panel studio-controls-panel">
          <button className="collapse-handle" type="button" onClick={() => ui.toggleStudioControls()} title={ui.studioControlsCollapsed ? 'Show controls' : 'Hide controls'}>
            {ui.studioControlsCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
          <div className="button-grid">
            <button className={generation.activity === 'txt2img' ? 'active' : ''} type="button" onClick={() => generation.setActivity('txt2img')}>
              txt2img
            </button>
            <button className={generation.activity === 'img2img' ? 'active' : ''} type="button" onClick={() => generation.setActivity('img2img')}>
              img2img
            </button>
          </div>

          <label className="field">
            <span>Preset</span>
            <select value={generation.preset} onChange={(event) => generation.setPreset(event.currentTarget.value)}>
              <option>Anima</option>
              <option>SD</option>
              <option>XL</option>
              <option>SDXL</option>
              <option>Flux</option>
              <option>Qwen</option>
              <option>ZImageTurbo</option>
              <option>Lumina</option>
              <option>Wan</option>
              <option>LTX</option>
            </select>
          </label>

          <label className="field">
            <span>Model</span>
            <select
              value={generation.modelPath}
              onChange={(event) => {
                const selected = models.find((model) => model.value === event.currentTarget.value);
                generation.setModel(event.currentTarget.value, selected?.name || '');
              }}
            >
              <option value="">Automatic</option>
              {models.map((model) => (
                <option key={model.value} value={model.value}>
                  {model.label}
                </option>
              ))}
            </select>
          </label>

          <div className="compact-note">
            Workflow: {generation.workflowName || 'Default backend route'}
          </div>

          <label className="field">
            <span>Prompt</span>
            <textarea value={generation.prompt} onChange={(event) => generation.setPrompt(event.currentTarget.value)} placeholder="Describe the image..." />
          </label>

          <label className="field">
            <span>Negative Prompt</span>
            <textarea value={generation.negativePrompt} onChange={(event) => generation.setNegativePrompt(event.currentTarget.value)} placeholder="Avoid..." />
          </label>

          {generation.activity === 'img2img' && (
            <div className="img2img-source">
              <div className="segmented-control compact-control">
                <button className={generation.img2imgMode === 'image' ? 'active' : ''} type="button" onClick={() => generation.setImg2ImgMode('image')}>
                  Image
                </button>
                <button className={generation.img2imgMode === 'inpaint' ? 'active' : ''} type="button" onClick={() => generation.setImg2ImgMode('inpaint')}>
                  Inpaint
                </button>
              </div>
              <label className="dropzone small-dropzone">
                <input
                  type="file"
                  accept="image/*"
                  onChange={async (event) => {
                    const file = event.currentTarget.files?.[0];
                    if (!file) return;
                    generation.setReferenceImage(await readFileAsDataUrl(file), file.name);
                  }}
                />
                <span>{generation.referenceImageName || 'Select reference image'}</span>
              </label>
              {generation.referenceImage && generation.img2imgMode === 'image' && <img src={generation.referenceImage} alt="img2img reference" />}
              {generation.referenceImage && generation.img2imgMode === 'inpaint' && (
                <InpaintCanvas image={generation.referenceImage} brushSize={generation.brushSize} onBrushSizeChange={generation.setBrushSize} onMaskChange={generation.setInpaintMaskImage} />
              )}
              <label className="field">
                <span>Denoise {generation.denoise.toFixed(2)}</span>
                <input type="range" min={0.05} max={1} step={0.01} value={generation.denoise} onChange={(event) => generation.setDenoise(Number(event.currentTarget.value))} />
              </label>
              <label className="field">
                <span>Resize</span>
                <select value={generation.resizeMode} onChange={(event) => generation.setResizeMode(event.currentTarget.value)}>
                  <option>Just Resize</option>
                  <option>Crop and Resize</option>
                  <option>Resize and Fill</option>
                  <option>Latent Upscale 2x</option>
                </select>
              </label>
              {generation.img2imgMode === 'inpaint' && (
                <div className="two-col">
                  <label className="field">
                    <span>Mask Blur</span>
                    <input type="number" min={0} max={64} value={generation.maskBlur} onChange={(event) => generation.setMaskBlur(Number(event.currentTarget.value))} />
                  </label>
                  <label className="field">
                    <span>Fill</span>
                    <select value={generation.maskContent} onChange={(event) => generation.setMaskContent(event.currentTarget.value)}>
                      <option>Original</option>
                    </select>
                  </label>
                </div>
              )}
            </div>
          )}

          <section className="controlnet-panel">
            <div className="control-row">
              <span>ControlNet</span>
              <button
                className={generation.controlNetEnabled ? 'toggle active' : 'toggle'}
                type="button"
                onClick={() => generation.setControlNetEnabled(!generation.controlNetEnabled)}
                disabled={!controlNetCompatible}
                aria-label="Toggle ControlNet"
              >
                <span />
              </button>
            </div>
            {generation.controlNetEnabled && controlNetCompatible && (
              <div className="control-stack">
                <div className="two-col">
                  <label className="field">
                    <span>Type</span>
                    <select value={generation.controlNetType} onChange={(event) => generation.setControlNetType(event.currentTarget.value)}>
                      <option value="canny">Canny</option>
                      <option value="depth">Depth</option>
                      <option value="openpose">OpenPose</option>
                      <option value="lineart">Lineart</option>
                      <option value="tile">Tile</option>
                      <option value="ltx_ic">LTX IC-LoRA</option>
                    </select>
                  </label>
                  <label className="field">
                    <span>Balance</span>
                    <select value={generation.controlNetBalance} onChange={(event) => generation.setControlNetBalance(event.currentTarget.value)}>
                      <option>Balanced</option>
                      <option>Control priority</option>
                      <option>Prompt priority</option>
                    </select>
                  </label>
                </div>
                <label className="field">
                  <span>Model</span>
                  <select
                    value={generation.controlNetModel}
                    onChange={(event) => {
                      const selected = controlNetModels.find((model) => model.value === event.currentTarget.value);
                      generation.setControlNetModel(event.currentTarget.value, selected?.name || 'Automatic');
                    }}
                  >
                    <option value="Automatic">Automatic</option>
                    {controlNetModels.map((model) => (
                      <option key={model.value} value={model.value}>
                        {model.label}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="dropzone small-dropzone">
                  <input
                    type="file"
                    accept="image/*"
                    onChange={async (event) => {
                      const file = event.currentTarget.files?.[0];
                      if (!file) return;
                      generation.setControlNetImage(await readFileAsDataUrl(file), file.name);
                    }}
                  />
                  <span>{generation.controlNetImageName || (generation.referenceImage ? 'Using img2img reference if empty' : 'Select control image')}</span>
                </label>
                {generation.controlNetImage && <img className="controlnet-thumb" src={generation.controlNetImage} alt="ControlNet source" />}
                <label className="field">
                  <span>Strength {generation.controlNetStrength.toFixed(2)}</span>
                  <input type="range" min={0} max={2} step={0.01} value={generation.controlNetStrength} onChange={(event) => generation.setControlNetStrength(Number(event.currentTarget.value))} />
                </label>
                <div className="two-col">
                  <label className="field">
                    <span>Start</span>
                    <input type="number" min={0} max={1} step={0.05} value={generation.controlNetStart} onChange={(event) => generation.setControlNetRange(Number(event.currentTarget.value), generation.controlNetEnd)} />
                  </label>
                  <label className="field">
                    <span>End</span>
                    <input type="number" min={0} max={1} step={0.05} value={generation.controlNetEnd} onChange={(event) => generation.setControlNetRange(generation.controlNetStart, Number(event.currentTarget.value))} />
                  </label>
                </div>
                <p className="compact-note">
                  {controlNetModels.length ? `${controlNetModels.length} compatible model(s) detected.` : 'No compatible local model detected for this preset/type.'}
                </p>
              </div>
            )}
            {!controlNetCompatible && <p className="compact-note">ControlNet is available for SD, SDXL and LTX routes.</p>}
          </section>

          {videoMode && (
            <section className="video-panel">
              <div className="control-row">
                <span>{generation.preset} Video</span>
                <em>{alignedFrames} frames</em>
              </div>
              <div className="three-col">
                <label className="field">
                  <span>Frames</span>
                  <input type="number" min={1} value={generation.videoFrames} onChange={(event) => generation.setVideoTiming(Number(event.currentTarget.value), generation.videoFps, generation.videoSeconds)} />
                </label>
                <label className="field">
                  <span>FPS</span>
                  <input type="number" min={1} max={60} value={generation.videoFps} onChange={(event) => generation.setVideoTiming(generation.videoFrames, Number(event.currentTarget.value), generation.videoSeconds)} />
                </label>
                <label className="field">
                  <span>Seconds</span>
                  <input type="number" min={0.1} step={0.1} value={generation.videoSeconds} onChange={(event) => generation.setVideoTiming(generation.videoFrames, generation.videoFps, Number(event.currentTarget.value))} />
                </label>
              </div>
              <label className="field">
                <span>Motion Strength {generation.videoMotionStrength.toFixed(2)}</span>
                <input type="range" min={0} max={1.5} step={0.01} value={generation.videoMotionStrength} onChange={(event) => generation.setVideoMotionStrength(Number(event.currentTarget.value))} />
              </label>
              <div className="two-col">
                <label className="field">
                  <span>Video VAE</span>
                  <input value={generation.videoVae} onChange={(event) => generation.setVideoVae(event.currentTarget.value)} />
                </label>
                <label className="field">
                  <span>Audio VAE</span>
                  <input value={generation.audioVae} onChange={(event) => generation.setAudioVae(event.currentTarget.value)} />
                </label>
              </div>
              {generation.preset.toLowerCase() === 'ltx' && (
                <>
                  <div className="control-row">
                    <span>Active audio</span>
                    <button className={generation.videoActiveAudio ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setVideoActiveAudio(!generation.videoActiveAudio)} aria-label="Toggle active audio">
                      <span />
                    </button>
                  </div>
                  <label className="field">
                    <span>Latent Upscale</span>
                    <input value={generation.latentUpscale} onChange={(event) => generation.setLatentUpscale(event.currentTarget.value)} />
                  </label>
                  <div className="control-row">
                    <span>Latent upscale refine</span>
                    <button className={generation.latentUpscaleRefine ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setLatentUpscaleRefine(!generation.latentUpscaleRefine)} aria-label="Toggle latent upscale refine">
                      <span />
                    </button>
                  </div>
                  <div className="three-col">
                    <label className="field">
                      <span>Tiles X</span>
                      <input type="number" min={1} max={8} value={generation.decodeTilesX} onChange={(event) => generation.setDecodeTiles(Number(event.currentTarget.value), generation.decodeTilesY, generation.decodeOverlap)} />
                    </label>
                    <label className="field">
                      <span>Tiles Y</span>
                      <input type="number" min={1} max={8} value={generation.decodeTilesY} onChange={(event) => generation.setDecodeTiles(generation.decodeTilesX, Number(event.currentTarget.value), generation.decodeOverlap)} />
                    </label>
                    <label className="field">
                      <span>Overlap</span>
                      <input type="number" min={0} max={256} value={generation.decodeOverlap} onChange={(event) => generation.setDecodeTiles(generation.decodeTilesX, generation.decodeTilesY, Number(event.currentTarget.value))} />
                    </label>
                  </div>
                </>
              )}
              <p className="compact-note">{generation.preset === 'Wan' ? 'Wan I2V uses the img2img reference as the start frame.' : 'LTX img2video uses the img2img reference for the linear route.'}</p>
            </section>
          )}

          <div className="two-col">
            <label className="field">
              <span>Width</span>
              <input type="number" min={64} step={8} value={generation.width} onChange={(event) => generation.setSize(Number(event.currentTarget.value), generation.height)} />
            </label>
            <label className="field">
              <span>Height</span>
              <input type="number" min={64} step={8} value={generation.height} onChange={(event) => generation.setSize(generation.width, Number(event.currentTarget.value))} />
            </label>
          </div>

          <div className="two-col">
            <label className="field">
              <span>Steps</span>
              <input type="number" min={1} max={150} value={generation.steps} onChange={(event) => generation.setSteps(Number(event.currentTarget.value))} />
            </label>
            <label className="field">
              <span>CFG</span>
              <input type="number" min={0} max={30} step={0.1} value={generation.cfg} onChange={(event) => generation.setCfg(Number(event.currentTarget.value))} />
            </label>
          </div>

          <div className="two-col">
            <label className="field">
              <span>Sampler</span>
              <select value={generation.sampler} onChange={(event) => generation.setSampler(event.currentTarget.value)}>
                <option value="euler_ancestral">Euler Ancestral</option>
                <option value="euler_ancestral_cfg_pp">Euler Ancestral CFG++</option>
                <option value="euler">Euler</option>
                <option value="dpmpp_2m">DPM++ 2M</option>
                <option value="dpmpp_sde">DPM++ SDE</option>
              </select>
            </label>
            <label className="field">
              <span>Scheduler</span>
              <select value={generation.scheduler} onChange={(event) => generation.setScheduler(event.currentTarget.value)}>
                <option value="karras">Karras</option>
                <option value="normal">Normal</option>
                <option value="quadratic">Quadratic</option>
                <option value="simple">Simple</option>
                <option value="sgm_uniform">SGM Uniform</option>
              </select>
            </label>
          </div>

          <label className="field">
            <span>Seed</span>
            <input type="number" value={generation.seed} onChange={(event) => generation.setSeed(Number(event.currentTarget.value))} />
          </label>

          <section className="lora-panel">
            <div className="control-row">
              <span>LoRAs</span>
              <button className="mini-button" type="button" onClick={() => loraStore.clearLoras()} disabled={!loraStore.activeLoras.length}>
                Clear
              </button>
            </div>
            {loraStore.activeLoras.length > 0 && (
              <div className="active-lora-list">
                {loraStore.activeLoras.map((lora) => (
                  <div className="active-lora" key={lora.relative_name}>
                    <span title={lora.relative_name}>{lora.name}</span>
                    <input type="number" min={-2} max={2} step={0.05} value={lora.strength} onChange={(event) => loraStore.updateStrength(lora.relative_name, Number(event.currentTarget.value))} />
                    <button type="button" onClick={() => loraStore.removeLora(lora.relative_name)}>
                      Remove
                    </button>
                  </div>
                ))}
              </div>
            )}
            <label className="field">
              <span>Search LoRAs</span>
              <input value={loraSearch} onChange={(event) => setLoraSearch(event.currentTarget.value)} placeholder="Filter local LoRAs..." />
            </label>
            <div className="lora-results">
              {visibleLoras.map((lora) => {
                const relativeName = (lora.relative_path || lora.name).replace(/^loras[\\/]/i, '').replaceAll('/', '\\');
                const active = loraStore.activeLoras.some((item) => item.relative_name === relativeName);
                return (
                  <button
                    key={lora.relative_path || lora.path}
                    className={active ? 'active' : ''}
                    type="button"
                    onClick={() =>
                      loraStore.addLora({
                        name: lora.name,
                        relative_name: relativeName,
                        relative_path: lora.relative_path,
                        strength: 0.8,
                        strength_model: 0.8,
                        strength_clip: 0.8,
                      })
                    }
                  >
                    {lora.name}
                  </button>
                );
              })}
            </div>
          </section>
        </aside>

        <main className="surface preview-panel studio-preview">
          {previewUrl ? (
            previewIsVideo ? (
              <video className="extras-media" src={previewUrl} controls playsInline />
            ) : (
              <img className="extras-media" src={previewUrl} alt="Studio output preview" />
            )
          ) : (
            <div className="preview-empty">
              <Play size={38} />
              <p>Generated outputs will appear here.</p>
            </div>
          )}

          {processing && (
            <div className="job-overlay">
              <LoaderCircle className="spin" size={34} />
              <strong>{activeJob?.progress ?? 0}%</strong>
              <span>{activeJob?.message || 'Starting generation...'}</span>
            </div>
          )}
        </main>

        {ui.studioGalleryOpen && (
          <aside className="surface studio-gallery-panel">
            <div className="control-row">
              <span>Gallery</span>
              <button className="mini-button" type="button" onClick={() => ui.toggleStudioGalleryExpanded()} title={ui.studioGalleryExpanded ? 'Compact gallery' : 'Expand gallery'}>
                {ui.studioGalleryExpanded ? <Minimize2 size={14} /> : <Maximize2 size={14} />}
              </button>
            </div>
            <div className="studio-gallery-list">
              {(gallery.data ?? []).slice(0, 18).map((item) => {
                const itemUrl = outputUrl(item.image || item.thumb);
                const isVideo = /\.(mp4|mov|webm|mkv|avi)$/i.test(item.filename || item.path || '');
                return (
                  <button
                    className="studio-gallery-item"
                    key={item.relative_path || item.path}
                    type="button"
                    onClick={() => {
                      generation.setActivity('img2img');
                      generation.setReferenceImage(outputUrl(item.image), item.filename);
                    }}
                    title="Send to img2img"
                  >
                    {isVideo ? (
                      <span className="studio-gallery-video">
                        <Clapperboard size={18} />
                      </span>
                    ) : (
                      <img src={itemUrl} alt={item.title || item.filename} />
                    )}
                    <span>{item.filename}</span>
                    <Send size={13} />
                  </button>
                );
              })}
            </div>
          </aside>
        )}
      </div>

      <footer className="extras-action-bar">
        <div className="status-line">
          <span>{activeJob?.message || 'Studio generation ready'}</span>
          {(localError || activeJob?.status === 'failed') && <strong>{localError || activeJob?.error || activeJob?.message}</strong>}
          {activeJob?.status === 'completed' && <em>{generatedOutput?.filename || 'Generation completed'}</em>}
        </div>
        <div className="action-buttons">
          <button type="button" className="flat-button" onClick={() => cancelMutation.mutate()} disabled={!processing || !jobId}>
            <Square size={14} />
            Cancel
          </button>
          <button type="button" className="primary-button" onClick={generate} disabled={processing}>
            {processing ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />}
            Generate
          </button>
        </div>
      </footer>
    </section>
  );
}
