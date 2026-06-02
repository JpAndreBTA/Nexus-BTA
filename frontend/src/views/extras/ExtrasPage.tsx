import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { ImageUp, LoaderCircle, Scissors, Video, X } from 'lucide-react';

import { nexusApi } from '../../api/nexusClient';
import type { ExtrasJob } from '../../api/types';
import { useModelCatalogQuery } from '../../api/queries';
import { useExtrasStore } from '../../stores/extrasStore';
import { buildExtrasPlan, catalogNames, extrasPlanLabel, isImageFile, isVideoFile, sourceTypeForFiles, sourceTypeForState } from './extrasPlan';

const modeOptions = [
  { id: 'image', label: 'Image', icon: ImageUp },
  { id: 'video', label: 'Video', icon: Video },
  { id: 'remove_bg', label: 'Remove BG', icon: Scissors },
] as const;

const nvidiaUpscalers = ['nvidia_pid', 'nvidia_rtx'];

function outputUrl(url: string | undefined) {
  if (!url) return '';
  return url.startsWith('/') ? url : `/${url.replace(/^\/+/, '')}`;
}

function fileListLabel(files: File[]) {
  if (!files.length) return 'Drop or select source files';
  if (files.length === 1) return files[0].name;
  return `${files.length} frame sequence`;
}

function terminalStatus(job: ExtrasJob | undefined) {
  return job?.status === 'completed' || job?.status === 'failed' || job?.status === 'cancelled';
}

export function ExtrasPage() {
  const catalogQuery = useModelCatalogQuery();
  const catalog = catalogQuery.data;
  const extras = useExtrasStore();
  const [jobId, setJobId] = useState<string>('');
  const [localError, setLocalError] = useState<string>('');
  const [sourcePreviewUrl, setSourcePreviewUrl] = useState<string>('');
  const [isDragging, setIsDragging] = useState(false);

  const sourceType = useMemo(() => sourceTypeForState(extras), [extras]);
  const plan = useMemo(() => buildExtrasPlan(extras, catalog), [extras, catalog]);
  const planLabel = useMemo(() => extrasPlanLabel(plan), [plan]);

  const upscalers = useMemo(() => catalogNames(catalog, ['upscale_models'], ['Lanczos', ...nvidiaUpscalers]), [catalog]);
  const videoUpscalers = useMemo(() => catalogNames(catalog, ['upscale_models', 'latent_upscale_models'], ['Lanczos', ...nvidiaUpscalers]), [catalog]);
  const interpolators = useMemo(() => catalogNames(catalog, ['frame_interpolation'], ['frame_interpolation/rife_v4.26.safetensors']), [catalog]);
  const removeBgModels = useMemo(
    () => catalogNames(catalog, ['background_removal', 'RMBG'], ['BiRefNet'], (name) => /biref|rmbg|bria|inspy|isnet|ben|rembg|background/i.test(name)),
    [catalog],
  );

  const modelCounts = useMemo(
    () => ({
      upscalers: catalog?.categories.upscale_models?.length ?? 0,
      interpolators: catalog?.categories.frame_interpolation?.length ?? 0,
      backgroundRemoval: catalog?.categories.background_removal?.length ?? 0,
    }),
    [catalog],
  );

  useEffect(() => {
    if (extras.remoteSource) {
      setSourcePreviewUrl(extras.remoteSource.url);
      return;
    }
    if (!extras.files.length) {
      setSourcePreviewUrl('');
      return;
    }
    const url = URL.createObjectURL(extras.files[0]);
    setSourcePreviewUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [extras.files, extras.remoteSource]);

  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const source = params.get('source');
    if (!source) return;
    const mediaType = params.get('media') === 'video' ? 'video' : 'image';
    const title = params.get('title') || source.split('/').pop() || 'gallery source';
    extras.setRemoteSource({ url: source, title, mediaType });
    if (extras.mode !== 'remove_bg') extras.setMode(mediaType);
    window.history.replaceState(null, '', window.location.pathname);
    // URL import is intentionally one-shot when the route opens.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (upscalers.length && !upscalers.includes(extras.imageUpscaler)) extras.setImageUpscaler(upscalers[0]);
  }, [extras, upscalers]);

  useEffect(() => {
    if (videoUpscalers.length && !videoUpscalers.includes(extras.videoUpscaleModel)) extras.setVideoUpscaleModel(videoUpscalers[0]);
  }, [extras, videoUpscalers]);

  useEffect(() => {
    if (interpolators.length && !interpolators.includes(extras.interpolationModel)) extras.setInterpolationModel(interpolators[0]);
  }, [extras, interpolators]);

  useEffect(() => {
    if (removeBgModels.length && !removeBgModels.includes(extras.removeBgModel)) extras.setRemoveBgModel(removeBgModels[0]);
  }, [extras, removeBgModels]);

  const startMutation = useMutation({
    mutationFn: () => nexusApi.startExtras(plan.mode, plan, extras.files),
    onSuccess: (job) => {
      setJobId(job.job_id);
      setLocalError('');
    },
    onError: (error) => {
      setLocalError(error instanceof Error ? error.message : 'Extras failed to start.');
    },
  });

  const jobQuery = useQuery({
    queryKey: ['extras-job', jobId],
    queryFn: () => nexusApi.extrasJob(jobId),
    enabled: !!jobId,
    refetchInterval: (query) => {
      const job = query.state.data as ExtrasJob | undefined;
      return job && terminalStatus(job) ? false : 900;
    },
  });

  const activeJob = jobQuery.data ?? startMutation.data;
  const firstOutput = activeJob?.outputs?.[0];
  const processedUrl = activeJob?.status === 'completed' ? outputUrl(firstOutput?.url) : '';
  const processing = startMutation.isPending || (!!activeJob && !terminalStatus(activeJob));
  const previewKind = firstOutput?.kind || (sourceType === 'video' ? 'video' : 'image');
  const previewUrl = processedUrl || sourcePreviewUrl;

  function setFiles(files: File[]) {
    const valid = files.filter((file) => isImageFile(file) || isVideoFile(file));
    if (!valid.length) return;
    if (extras.mode !== 'remove_bg') {
      const nextType = sourceTypeForFiles(valid);
      extras.setMode(nextType === 'image' ? 'image' : 'video');
    }
    setJobId('');
    setLocalError('');
    extras.setFiles(valid);
  }

  function clearSource() {
    extras.clearSource();
    setJobId('');
    setLocalError('');
  }

  async function processExtras() {
    if (!extras.files.length && !extras.remoteSource) {
      setLocalError('Load an image, video, or frame sequence first.');
      return;
    }
    const nvidiaEngine = plan.upscale?.enabled && nvidiaUpscalers.includes(plan.upscale.engine || '') ? plan.upscale.engine || '' : '';
    if (nvidiaEngine) {
      const status = await nexusApi.nvidiaExtrasStatus(nvidiaEngine).catch(() => null);
      const label = status?.label || (nvidiaEngine === 'nvidia_pid' ? 'NVIDIA PiD Pixel Diffusion' : 'NVIDIA RTX Video Super Resolution');
      if (status?.installed && nvidiaEngine === 'nvidia_pid' && sessionStorage.getItem('nexus_pid_upscale_ack') !== '1') {
        const ok = window.confirm(`${label} is installed, but it is a latent decoder/upscaler and its ComfyUI node can download large PiD source/checkpoints/assets when auto_download is enabled. Continue with this PiD plan?`);
        if (!ok) return;
        sessionStorage.setItem('nexus_pid_upscale_ack', '1');
      }
      if (!status?.installed) {
        const missing = [
          status?.node_ready === false ? 'custom node' : '',
          status?.dependency_ready === false ? 'Python dependencies' : '',
        ].filter(Boolean).join(' and ') || 'runtime setup';
        const piDNote = nvidiaEngine === 'nvidia_pid'
          ? ' PiD is a latent decoder/upscaler and may download large checkpoints/assets only when its ComfyUI node runs with auto_download.'
          : ' RTX Video Super Resolution needs an RTX GPU plus NVIDIA VFX runtime support.';
        if (!window.confirm(`${label} is selected but ${missing} is not fully ready.${piDNote} Continue using the standard Lanczos fallback for this run?`)) return;
      }
    }
    setLocalError('');
    setJobId('');
    await startMutation.mutateAsync();
  }

  const accept = extras.mode === 'image' ? 'image/*' : extras.mode === 'video' ? 'video/*,image/*' : 'image/*,video/*';

  return (
    <section className="page extras-layout">
      <header className="page-controls">
        <span className="route-sentinel" data-route="extras" aria-label="Extras workspace" />
        <div className="segmented-control">
          {modeOptions.map((option) => {
            const Icon = option.icon;
            return (
              <button key={option.id} className={extras.mode === option.id ? 'active' : ''} type="button" onClick={() => extras.setMode(option.id)}>
                <Icon size={14} />
                {option.label}
              </button>
            );
          })}
        </div>
      </header>

      <div className="extras-columns">
        <aside className="surface tool-panel">
          <label
            className={isDragging ? 'dropzone active' : 'dropzone'}
            onDragOver={(event) => {
              event.preventDefault();
              setIsDragging(true);
            }}
            onDragLeave={() => setIsDragging(false)}
            onDrop={(event) => {
              event.preventDefault();
              setIsDragging(false);
              setFiles(Array.from(event.dataTransfer.files));
            }}
          >
            <input
              multiple={extras.mode !== 'image'}
              type="file"
              accept={accept}
              onChange={(event) => setFiles(Array.from(event.currentTarget.files ?? []))}
            />
            <span title={extras.remoteSource?.title || fileListLabel(extras.files)}>{extras.remoteSource?.title || fileListLabel(extras.files)}</span>
          </label>

          <div className="model-stats">
            <span>{modelCounts.upscalers} upscale</span>
            <span>{modelCounts.interpolators} interpolate</span>
            <span>{modelCounts.backgroundRemoval} remove bg</span>
          </div>

          <div className="extras-preset-strip">
            {extras.mode === 'image' && (
              <>
                <button type="button" onClick={() => { extras.setImageScale('2x'); extras.setExportFormat('png'); }}>Fast 2x PNG</button>
                <button type="button" onClick={() => { extras.setImageScale('4x'); extras.setPreserveAlpha(true); }}>Alpha 4x</button>
              </>
            )}
            {extras.mode === 'video' && (
              <>
                <button type="button" onClick={() => { extras.setVideoInterpolateEnabled(true); extras.setTargetFps(60); extras.setEncoder('mp4_h264'); }}>60 FPS MP4</button>
                <button type="button" onClick={() => { extras.setVideoUpscaleEnabled(true); extras.setVideoScale('2x'); }}>2x Video</button>
              </>
            )}
            {extras.mode === 'remove_bg' && (
              <>
                <button type="button" onClick={() => { extras.setRemoveBgPreset('image'); extras.setRemoveBgOutput('rgba'); extras.setPreserveAlpha(true); }}>RGBA Cutout</button>
                <button type="button" onClick={() => { extras.setRemoveBgPreset('video'); extras.setRemoveBgOutput('both'); }}>Video Matte</button>
              </>
            )}
          </div>

          {extras.mode === 'image' && (
            <div className="control-stack">
              <label className="field">
                <span>Upscale model</span>
                <select value={extras.imageUpscaler} onChange={(event) => extras.setImageUpscaler(event.currentTarget.value)}>
                  {upscalers.map((model) => (
                    <option key={model}>{model}</option>
                  ))}
                </select>
              </label>
              <div className="button-grid">
                {(['2x', '4x', 'custom'] as const).map((scale) => (
                  <button key={scale} className={extras.imageScale === scale ? 'active' : ''} type="button" onClick={() => extras.setImageScale(scale)}>
                    {scale}
                  </button>
                ))}
              </div>
              {extras.imageScale === 'custom' && (
                <div className="two-col">
                  <label className="field">
                    <span>Width</span>
                    <input type="number" min={64} value={extras.customWidth} onChange={(event) => extras.setCustomSize(Number(event.currentTarget.value), extras.customHeight)} />
                  </label>
                  <label className="field">
                    <span>Height</span>
                    <input type="number" min={64} value={extras.customHeight} onChange={(event) => extras.setCustomSize(extras.customWidth, Number(event.currentTarget.value))} />
                  </label>
                </div>
              )}
              <label className="field">
                <span>Export</span>
                <select value={extras.exportFormat} onChange={(event) => extras.setExportFormat(event.currentTarget.value as 'png' | 'webp' | 'jpg')}>
                  <option value="png">PNG</option>
                  <option value="webp">WEBP</option>
                  <option value="jpg">JPG</option>
                </select>
              </label>
            </div>
          )}

          {extras.mode === 'video' && (
            <div className="control-stack">
              {sourceType === 'image_sequence' && (
                <label className="field">
                  <span>Source FPS</span>
                  <input type="number" min={1} max={240} value={extras.sourceFps} onChange={(event) => extras.setSourceFps(Number(event.currentTarget.value))} />
                </label>
              )}
              <div className="control-row">
                <span>Interpolate</span>
                <button className={extras.videoInterpolateEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => extras.setVideoInterpolateEnabled(!extras.videoInterpolateEnabled)}>
                  <span />
                </button>
              </div>
              {extras.videoInterpolateEnabled && (
                <>
                  <label className="field">
                    <span>Interpolation model</span>
                    <select value={extras.interpolationModel} onChange={(event) => extras.setInterpolationModel(event.currentTarget.value)}>
                      {interpolators.map((model) => (
                        <option key={model}>{model}</option>
                      ))}
                    </select>
                  </label>
                  <label className="field">
                    <span>Target FPS</span>
                    <input type="number" min={1} max={240} value={extras.targetFps} onChange={(event) => extras.setTargetFps(Number(event.currentTarget.value))} />
                  </label>
                </>
              )}
              <div className="control-row">
                <span>Video upscale</span>
                <button className={extras.videoUpscaleEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => extras.setVideoUpscaleEnabled(!extras.videoUpscaleEnabled)}>
                  <span />
                </button>
              </div>
              {extras.videoUpscaleEnabled && (
                <>
                  <label className="field">
                    <span>Upscale model</span>
                    <select value={extras.videoUpscaleModel} onChange={(event) => extras.setVideoUpscaleModel(event.currentTarget.value)}>
                      {videoUpscalers.map((model) => (
                        <option key={model}>{model}</option>
                      ))}
                    </select>
                  </label>
                  <div className="two-col">
                    <label className="field">
                      <span>Scale</span>
                      <select value={extras.videoScale} onChange={(event) => extras.setVideoScale(event.currentTarget.value as '2x' | '4x')}>
                        <option value="2x">2x</option>
                        <option value="4x">4x</option>
                      </select>
                    </label>
                    <label className="field">
                      <span>Tile</span>
                      <input type="number" min={128} step={64} value={extras.tileSize} onChange={(event) => extras.setTileSize(Number(event.currentTarget.value))} />
                    </label>
                  </div>
                </>
              )}
              <label className="field">
                <span>Encoder</span>
                <select value={extras.encoder} onChange={(event) => extras.setEncoder(event.currentTarget.value as typeof extras.encoder)}>
                  <option value="mp4_h264">MP4 H.264</option>
                  <option value="webm_vp9">WEBM VP9</option>
                  <option value="mov_prores_422">MOV ProRes 422</option>
                  <option value="mov_prores_4444_alpha">MOV ProRes 4444 Alpha</option>
                  <option value="image_sequence_png">PNG sequence</option>
                  <option value="image_sequence_png_alpha">PNG alpha sequence</option>
                </select>
              </label>
            </div>
          )}

          {extras.mode === 'remove_bg' && (
            <div className="control-stack">
              <div className="button-grid">
                <button className={extras.removeBgPreset === 'image' ? 'active' : ''} type="button" onClick={() => extras.setRemoveBgPreset('image')}>
                  Image
                </button>
                <button className={extras.removeBgPreset === 'video' ? 'active' : ''} type="button" onClick={() => extras.setRemoveBgPreset('video')}>
                  Video
                </button>
              </div>
              <label className="field">
                <span>Background model</span>
                <select value={extras.removeBgModel} onChange={(event) => extras.setRemoveBgModel(event.currentTarget.value)}>
                  {removeBgModels.map((model) => (
                    <option key={model}>{model}</option>
                  ))}
                </select>
              </label>
              <label className="field">
                <span>Output</span>
                <select value={extras.removeBgOutput} onChange={(event) => extras.setRemoveBgOutput(event.currentTarget.value as 'rgba' | 'mask' | 'both')}>
                  <option value="rgba">RGBA</option>
                  <option value="mask">Mask</option>
                  <option value="both">Both</option>
                </select>
              </label>
              <label className="field">
                <span>Threshold {extras.removeBgThreshold.toFixed(2)}</span>
                <input type="range" min={0} max={1} step={0.01} value={extras.removeBgThreshold} onChange={(event) => extras.setRemoveBgThreshold(Number(event.currentTarget.value))} />
              </label>
            </div>
          )}

          <div className="control-row">
            <span>Preserve alpha</span>
            <button className={extras.preserveAlpha ? 'toggle active' : 'toggle'} type="button" onClick={() => extras.setPreserveAlpha(!extras.preserveAlpha)} aria-pressed={extras.preserveAlpha}>
              <span />
            </button>
          </div>
        </aside>

        <main className="surface preview-panel">
          {previewUrl ? (
            previewKind === 'video' || sourceType === 'video' ? (
              <video className="extras-media" src={previewUrl} controls playsInline />
            ) : (
              <img className="extras-media" src={previewUrl} alt="Extras preview" />
            )
          ) : (
            <div className="preview-empty">
              <ImageUp size={38} />
              <p>Load a source to preview and process Extras.</p>
            </div>
          )}

          {processing && (
            <div className="job-overlay">
              <LoaderCircle className="spin" size={34} />
              <strong>{activeJob?.progress ?? 0}%</strong>
              <span>{activeJob?.message || 'Starting Extras job...'}</span>
            </div>
          )}
        </main>
      </div>

      <footer className="extras-action-bar">
        <div className="status-line">
          <span>{extras.files.length || extras.remoteSource ? `${sourceType} / ${planLabel}` : 'Extras pipeline ready'}</span>
          {(localError || activeJob?.status === 'failed') && <strong>{localError || activeJob?.error || activeJob?.message}</strong>}
          {activeJob?.status === 'completed' && <em>{firstOutput?.filename || 'Extras completed'}</em>}
        </div>
        <div className="action-buttons">
          <button type="button" className="flat-button" onClick={clearSource} disabled={!extras.files.length && !extras.remoteSource && !jobId}>
            <X size={14} />
            Clear
          </button>
          <button type="button" className="primary-button" onClick={processExtras} disabled={processing || (!extras.files.length && !extras.remoteSource)}>
            {processing ? <LoaderCircle className="spin" size={14} /> : <ImageUp size={14} />}
            Process
          </button>
        </div>
      </footer>
    </section>
  );
}
