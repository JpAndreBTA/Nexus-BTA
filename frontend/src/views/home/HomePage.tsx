import { useEffect, useMemo, useState } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { LoaderCircle, Play, Server, Square } from 'lucide-react';

import { nexusApi } from '../../api/nexusClient';
import type { GenerateRequest, GenerationJob } from '../../api/types';
import { useGalleryQuery, useHealthQuery, useModelCatalogQuery } from '../../api/queries';
import { useLorasQuery } from '../../api/queries';
import { queryClient } from '../../shared/queryClient';
import { useGenerationStore } from '../../stores/generationStore';
import { useLoraStore } from '../../stores/loraStore';

function terminalStatus(job: GenerationJob | undefined) {
  return job?.status === 'completed' || job?.status === 'failed' || job?.status === 'cancelled';
}

function outputUrl(url: string | undefined) {
  if (!url) return '';
  return url.startsWith('/') ? url : `/${url.replace(/^\/+/, '')}`;
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

export function HomePage() {
  const health = useHealthQuery();
  const catalog = useModelCatalogQuery();
  const gallery = useGalleryQuery();
  const loras = useLorasQuery();
  const generation = useGenerationStore();
  const loraStore = useLoraStore();
  const [jobId, setJobId] = useState('');
  const [localError, setLocalError] = useState('');
  const [loraSearch, setLoraSearch] = useState('');

  const allModels = useMemo(() => modelOptions(catalog.data), [catalog.data]);
  const models = useMemo(() => {
    const filtered = allModels.filter((model) => modelMatchesPreset(model, generation.preset));
    return filtered.length ? filtered : allModels;
  }, [allModels, generation.preset]);
  const newestGalleryItem = gallery.data?.[0];
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

  const payload = useMemo<GenerateRequest>(
    () => ({
      activity: 'txt2img',
      workspace: 'viewer',
      preset: generation.preset,
      workflow_id: null,
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
      denoise: 1,
      vae: 'Automatic',
      text_encoder: 'Automatic',
      loras: loraStore.activeLoras,
      distilled_loras: [],
      video: {},
    }),
    [generation, loraStore.activeLoras],
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
    setLocalError('');
    setJobId('');
    await startMutation.mutateAsync();
  }

  return (
    <section className="page studio-layout">
      <header className="page-header">
        <div>
          <p className="eyebrow">Nexus BTA Web App</p>
          <h1>Studio</h1>
        </div>
        <span className={health.data?.nexus === 'ok' ? 'status-pill ok' : 'status-pill'}>
          <Server size={14} />
          {health.data?.nexus === 'ok' ? 'Backend online' : 'Backend check'}
        </span>
      </header>

      <div className="studio-columns">
        <aside className="surface tool-panel">
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

          <label className="field">
            <span>Prompt</span>
            <textarea value={generation.prompt} onChange={(event) => generation.setPrompt(event.currentTarget.value)} placeholder="Describe the image..." />
          </label>

          <label className="field">
            <span>Negative Prompt</span>
            <textarea value={generation.negativePrompt} onChange={(event) => generation.setNegativePrompt(event.currentTarget.value)} placeholder="Avoid..." />
          </label>

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
