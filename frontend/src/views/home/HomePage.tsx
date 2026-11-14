import { useEffect, useMemo, useRef, useState, type PointerEvent, type WheelEvent } from 'react';
import { useMutation, useQuery } from '@tanstack/react-query';
import { Brush, Clapperboard, FilePlus2, Grid3X3, Images, LoaderCircle, Maximize2, Minimize2, PanelLeftClose, PanelLeftOpen, Play, Redo2, Save, Send, SlidersHorizontal, Undo2, Wand2, Workflow, X } from 'lucide-react';

import { nexusApi } from '../../api/nexusClient';
import type { CatalogAsset, GenerateRequest, GenerationJob, Ideogram4PromptJsonRequest } from '../../api/types';
import { useWorkflowCapabilitiesQuery, useWorkflowInputReportQuery, useGalleryQuery, useModelCatalogQuery, useWorkflowAnalysisQuery, useWorkflowsQuery } from '../../api/queries';
import { useLorasQuery } from '../../api/queries';
import type { WorkflowGraphLink, WorkflowGraphNode, WorkflowSummary } from '../../api/types';
import { useGenerationStore, type GenerationState, type PromptRegion } from '../../stores/generationStore';
import { useLoraStore } from '../../stores/loraStore';
import { useUiStore } from '../../stores/uiStore';
import { InpaintCanvas } from './InpaintCanvas';
import { StudioNodeCapabilityStatus } from './StudioNodeCapabilityStatus';
import { StudioWorkflowInputStatus } from './StudioWorkflowInputStatus';
import { buildGenerationPayload, validateGeneration } from './generationPayload';
import { IdeogramRegionEditor } from './IdeogramRegionEditor';
import { StudioWorkflowGraph, buildStudioWorkflowGraph, music3WorkflowGraph, preferredWorkflow } from './StudioWorkflowGraph';
import { alignVideoFrames, assetSelectOptions, controlNetCompatiblePreset, controlNetModelOptions, modelOptions, primaryModelMatchesPreset, videoPreset } from './StudioCatalog';

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

function formatBytes(value: number | undefined) {
  const bytes = Number(value || 0);
  if (!bytes) return '0 B';
  const units = ['B', 'KB', 'MB', 'GB', 'TB'];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / 1024 ** index).toFixed(index >= 3 ? 2 : 1)} ${units[index]}`;
}

async function readFilesAsReferenceImages(files: FileList | File[]) {
  const imageFiles = Array.from(files).filter((file) => file.type.startsWith('image/'));
  return Promise.all(imageFiles.map(async (file) => ({ dataUrl: await readFileAsDataUrl(file), name: file.name })));
}

export function HomePage() {
  const catalog = useModelCatalogQuery();
  const gallery = useGalleryQuery();
  const loras = useLorasQuery();
  const workflows = useWorkflowsQuery();
  const generation = useGenerationStore();
  const loraStore = useLoraStore();
  const ui = useUiStore();
  const [jobId, setJobId] = useState('');
  const [localError, setLocalError] = useState('');
  const [loraSearch, setLoraSearch] = useState('');
  const [loraModalOpen, setLoraModalOpen] = useState(false);
  const [viewMode, setViewMode] = useState<'director' | 'linear' | 'inpaint' | 'workflow'>('linear');
  const [linearZoom, setLinearZoom] = useState(1);
  const [ideogramAssetSelection, setIdeogramAssetSelection] = useState<string[]>([]);
  const [ideogramPromptProvider, setIdeogramPromptProvider] = useState<Ideogram4PromptJsonRequest['provider']>('comfy_gemma4');
  const [ideogramPromptModel, setIdeogramPromptModel] = useState('gemma4:e2b');
  const [ideogramPromptEndpoint, setIdeogramPromptEndpoint] = useState('');
  const [ideogramPromptMessage, setIdeogramPromptMessage] = useState('');

  const allModels = useMemo(() => modelOptions(catalog.data), [catalog.data]);
  const vaeOptions = useMemo(() => assetSelectOptions(catalog.data, ['vae']), [catalog.data]);
  const textEncoderOptions = useMemo(() => assetSelectOptions(catalog.data, ['text_encoders', 'clip']), [catalog.data]);
  const audioVaeOptions = useMemo(() => assetSelectOptions(catalog.data, ['vae'], ['audio', 'ltx']), [catalog.data]);
  const latentUpscaleOptions = useMemo(() => assetSelectOptions(catalog.data, ['latent_upscale_models', 'upscale_models'], ['ltx', 'spatial']), [catalog.data]);
  const studioWorkflow = useMemo(() => preferredWorkflow(workflows.data, generation.preset, generation.workflowId), [generation.preset, generation.workflowId, workflows.data]);
  const workflowAnalysis = useWorkflowAnalysisQuery(studioWorkflow?.id || '');
  const workflowClassCount = studioWorkflow?.class_types?.length ?? 0;
  const nodeCapabilities = useWorkflowCapabilitiesQuery(studioWorkflow?.id || '', viewMode === 'workflow');
  const workflowInputReport = useWorkflowInputReportQuery(studioWorkflow?.id || '', viewMode === 'workflow');
  const models = useMemo(() => {
    const filtered = allModels.filter((model) => primaryModelMatchesPreset(model, generation.preset));
    if (['ideogram4', 'ideogram', 'krea2', 'krea-2'].includes(generation.preset.toLowerCase())) return filtered;
    return filtered.length ? filtered : allModels;
  }, [allModels, generation.preset]);
  const newestGalleryItem = gallery.data?.[0];
  const controlNetCompatible = controlNetCompatiblePreset(generation.preset);
  const videoMode = videoPreset(generation.preset);
  const qwenImageEdit = generation.preset.toLowerCase() === 'qwen';
  const ideogram4Mode = ['ideogram4', 'ideogram'].includes(generation.preset.toLowerCase());
  const krea2Mode = ['krea2', 'krea-2'].includes(generation.preset.toLowerCase());
  const music3Mode = ['music3', 'music 3', 'minimax_music3', 'minimax-music-3'].includes(generation.preset.toLowerCase());
  const ltxDirectorView = generation.preset.toLowerCase() === 'ltx' && viewMode === 'director';
  const directorMode = generation.preset.toLowerCase() === 'ltx' && generation.directorEnabled;
  const alignedFrames = alignVideoFrames(generation.preset, generation.videoFrames);
  const allReferenceImages = useMemo(
    () => Array.from(new Set(generation.referenceImage ? [generation.referenceImage, ...generation.extraReferenceImages.map((image) => image.dataUrl)] : [])).slice(0, 3),
    [generation.extraReferenceImages, generation.referenceImage],
  );
  const directorFrames = alignVideoFrames('ltx', Math.round(generation.directorDuration * generation.videoFps));
  const controlNetModels = useMemo(() => controlNetModelOptions(catalog.data, generation.preset, generation.controlNetType), [catalog.data, generation.preset, generation.controlNetType]);
  const ideogram4Status = useQuery({
    queryKey: ['ideogram4-assets-status'],
    queryFn: nexusApi.ideogram4AssetsStatus,
    enabled: ideogram4Mode,
    refetchInterval: ideogram4Mode ? 5000 : false,
  });
  const ideogram4Download = useMutation({
    mutationFn: (assets: string[]) => nexusApi.startIdeogram4AssetDownload({ assets, install_node_dependencies: true }),
    onSuccess: () => {
      void ideogram4Status.refetch();
    },
    onError: (error) => setLocalError(error instanceof Error ? error.message : 'Ideogram 4 dependency download failed.'),
  });
  const krea2Status = useQuery({
    queryKey: ['krea2-assets-status'],
    queryFn: nexusApi.krea2AssetsStatus,
    enabled: krea2Mode,
    refetchInterval: krea2Mode ? 5000 : false,
  });
  const krea2Download = useMutation({
    mutationFn: (assets: string[]) => nexusApi.startKrea2AssetDownload({ assets }),
    onSuccess: () => { void krea2Status.refetch(); },
    onError: (error) => setLocalError(error instanceof Error ? error.message : 'Krea 2 dependency download failed.'),
  });
  const music3Status = useQuery({
    queryKey: ['music3-assets-status'],
    queryFn: nexusApi.music3AssetsStatus,
    enabled: music3Mode,
    refetchInterval: music3Mode ? 5000 : false,
  });
  const music3Download = useMutation({
    mutationFn: (assets: string[]) => nexusApi.startMusic3AssetDownload({ assets }),
    onSuccess: () => { void music3Status.refetch(); void catalog.refetch(); },
    onError: (error) => setLocalError(error instanceof Error ? error.message : 'Music 3 dependency download failed.'),
  });
  const krea2PromptShown = useRef(false);
  const krea2ProfileApplied = useRef(false);
  const music3PromptShown = useRef(false);
  const ideogramPromptJson = useMutation({
    mutationFn: (request: Ideogram4PromptJsonRequest) => nexusApi.ideogram4PromptJson(request),
    onSuccess: (result) => {
      generation.setPrompt(result.prompt_text);
      setIdeogramPromptMessage(result.message || `JSON prompt generated with ${result.provider}.`);
      setLocalError('');
    },
    onError: (error) => {
      setIdeogramPromptMessage('');
      setLocalError(error instanceof Error ? error.message : 'Ideogram 4 Magic JSON prompt failed.');
    },
  });
  const ideogramAssets = ideogram4Status.data?.assets ?? [];
  const selectedIdeogramAssets = ideogramAssetSelection.length
    ? ideogramAssetSelection
    : (ideogram4Status.data?.missing_required_assets ?? []).map((asset) => asset.key);
  const studioWorkflowGraph = useMemo(() => {
    if (music3Mode) return music3WorkflowGraph(generation);
    const graph = workflowAnalysis.data?.visual_graph;
    return buildStudioWorkflowGraph(graph?.nodes ?? [], graph?.links ?? [], generation, alignedFrames, directorFrames);
  }, [alignedFrames, directorFrames, generation, workflowAnalysis.data?.visual_graph]);
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
    if (!krea2Mode) {
      krea2PromptShown.current = false;
      krea2ProfileApplied.current = false;
      return;
    }
    if (krea2PromptShown.current || krea2Status.isLoading || !krea2Status.data || krea2Status.data.generation_ready || !krea2Status.data.missing_required_assets?.length) return;
    krea2PromptShown.current = true;
    const missing = krea2Status.data.missing_required_assets ?? [];
    const answer = window.confirm(`Krea 2 não encontrou todos os modelos locais (${missing.map((item) => item.filename).join(', ')}). Deseja baixar os arquivos necessários agora para ${krea2Status.data.models_dir || 'a pasta de modelos configurada'}?`);
    if (answer) krea2Download.mutate(missing.map((item) => item.key));
  }, [krea2Download, krea2Mode, krea2Status.data, krea2Status.isLoading]);

  useEffect(() => {
    if (!krea2Mode || krea2ProfileApplied.current || !krea2Status.data?.recommended_profile) return;
    krea2ProfileApplied.current = true;
    generation.setSize(
      krea2Status.data.recommended_profile === 'rtx_5090' ? 1024 : 768,
      krea2Status.data.recommended_profile === 'rtx_5090' ? 1024 : 768,
    );
  }, [generation, krea2Mode, krea2Status.data?.recommended_profile]);

  useEffect(() => {
    if (!music3Mode) { music3PromptShown.current = false; return; }
    if (music3PromptShown.current || music3Status.isLoading || !music3Status.data || music3Status.data.generation_ready || (!music3Status.data.missing_required_assets?.length && !music3Status.data.missing_core_nodes?.length)) return;
    music3PromptShown.current = true;
    const missing = music3Status.data.missing_required_assets;
    const core = music3Status.data.missing_core_nodes?.length ? `\n\nO ComfyUI também precisa ser atualizado para fornecer: ${music3Status.data.missing_core_nodes.join(', ')}. Deseja baixar os modelos agora?` : '';
    if (window.confirm(`Music 3 encontrou dependências ausentes (${missing.map((item) => item.filename).join(', ') || 'nodes nativos do ComfyUI'}).${core}\n\nBaixar os modelos selecionados agora para ${music3Status.data.models_dir || 'a pasta configurada'}?`)) {
      if (missing.length) music3Download.mutate(missing.map((item) => item.key));
    }
  }, [music3Download, music3Mode, music3Status.data, music3Status.isLoading]);

  useEffect(() => {
    if ((!generation.modelPath || !models.some((model) => model.value === generation.modelPath)) && models[0]) {
      generation.setModel(models[0].value, models[0].name);
    } else if (krea2Mode && !models.length && generation.modelPath) {
      generation.setModel('', '');
    }
  }, [generation, krea2Mode, models]);

  useEffect(() => {
    if (!controlNetCompatible && generation.controlNetEnabled) {
      generation.setControlNetEnabled(false);
    }
    if (generation.controlNetModel !== 'Automatic' && !controlNetModels.some((model) => model.value === generation.controlNetModel)) {
      generation.setControlNetModel('Automatic', 'Automatic');
    }
  }, [controlNetCompatible, controlNetModels, generation]);

  useEffect(() => {
    if (!qwenImageEdit) return;
    if (generation.activity !== 'img2img') {
      generation.setActivity('img2img');
    }
    if (generation.img2imgMode !== 'image') {
      generation.setImg2ImgMode('image');
    }
  }, [generation, qwenImageEdit]);

  const payload = useMemo<GenerateRequest>(
    () => buildGenerationPayload({
      generation,
      activeLoras: loraStore.activeLoras,
      allReferenceImages,
      alignedFrames,
      directorFrames,
      controlNetCompatible,
      directorMode,
      ideogram4Mode,
      music3Mode,
      videoMode,
      music3RecommendedProfile: music3Status.data?.recommended_profile,
    }),
    [alignedFrames, allReferenceImages, controlNetCompatible, directorFrames, directorMode, generation, ideogram4Mode, loraStore.activeLoras, music3Mode, music3Status.data?.recommended_profile, videoMode],
  );

  const startMutation = useMutation({
    mutationFn: (request: GenerateRequest) => nexusApi.startGeneration(request),
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

  const activeJob = jobQuery.data ?? startMutation.data;
  const processing = startMutation.isPending || (!!activeJob && !terminalStatus(activeJob));
  const generatedOutput = activeJob?.status === 'completed' ? activeJob.outputs?.[0] : null;
  const previewUrl = outputUrl(generatedOutput?.url || (music3Mode ? '' : newestGalleryItem?.image));
  const previewIsVideo = /\.(mp4|mov|webm|mkv|avi)$/i.test(generatedOutput?.filename || newestGalleryItem?.filename || '');
  const previewIsAudio = generatedOutput?.kind === 'audio' || /\.(mp3|wav|flac|opus|m4a|ogg)$/i.test(generatedOutput?.filename || newestGalleryItem?.filename || '');

  function handleLinearViewerWheel(event: WheelEvent<HTMLDivElement>) {
    if (viewMode !== 'linear') return;
    event.preventDefault();
    setLinearZoom((value) => Math.max(0.25, Math.min(3, Number((value + (event.deltaY > 0 ? -0.08 : 0.08)).toFixed(2)))));
  }

  async function waitForIdeogramOllamaPull(jobId: string) {
    for (let attempt = 0; attempt < 720; attempt += 1) {
      const job = await nexusApi.ideogram4OllamaPullJob(jobId);
      setIdeogramPromptMessage(`${job.message}${Number.isFinite(job.progress) ? ` ${Math.round(job.progress)}%` : ''}`);
      if (['downloaded', 'completed'].includes(job.status)) return;
      if (job.status === 'failed') throw new Error(job.error || job.message || 'Ollama model download failed.');
      await new Promise((resolve) => window.setTimeout(resolve, 1500));
    }
    throw new Error('Ollama model download timed out.');
  }

  async function ensureIdeogramOllamaModel(model: string, endpoint: string) {
    const status = await nexusApi.ideogram4OllamaStatus(model, endpoint);
    if (!status.running) {
      const useFallback = window.confirm(`${status.message}\n\nUse the local template fallback instead?`);
      return useFallback ? 'template' : '';
    }
    if (status.installed) return 'ollama';
    const size = formatBytes(status.estimated_size_bytes);
    const download = window.confirm(`Ollama model "${model}" is not installed.\nEstimated download size: ${size}.\n\nDownload it now?`);
    if (!download) {
      const useFallback = window.confirm('Use the local template fallback without downloading a model?');
      return useFallback ? 'template' : '';
    }
    const job = await nexusApi.startIdeogram4OllamaPull({ model, endpoint });
    await waitForIdeogramOllamaPull(job.job_id);
    return 'ollama';
  }

  async function handleIdeogramMagicPrompt() {
    if (!(music3Mode ? (generation.musicCaption || generation.prompt) : generation.prompt).trim()) {
      setLocalError('Prompt is required.');
      return;
    }
    const choice = window.prompt(
      [
        'Choose Magic Prompt provider:',
        '1 - Comfy Gemma4 local',
        '2 - Gemma / Ollama local',
        '3 - Ideogram Magic Prompt API',
        '4 - OpenAI-compatible public API',
        '5 - Local template fallback',
      ].join('\n'),
      ideogramPromptProvider === 'ollama' ? '2' : ideogramPromptProvider === 'ideogram_magic' ? '3' : ideogramPromptProvider === 'openai_compatible' ? '4' : ideogramPromptProvider === 'template' ? '5' : '1',
    );
    if (choice === null) return;
    let provider: Ideogram4PromptJsonRequest['provider'] = 'template';
    let model = ideogramPromptModel;
    let endpoint = ideogramPromptEndpoint;
    const normalizedChoice = choice.trim().toLowerCase();
    if (['1', 'comfy', 'comfy_gemma4', 'gemma4fp8'].includes(normalizedChoice)) {
      provider = 'comfy_gemma4';
      model = '';
      endpoint = '';
    } else if (['2', 'gemma', 'ollama', 'local'].includes(normalizedChoice)) {
      provider = 'ollama';
      model = window.prompt('Ollama model name:', model || 'gemma4:e2b')?.trim() || '';
      if (!model) return;
      endpoint = window.prompt('Ollama endpoint:', endpoint || 'http://127.0.0.1:11434')?.trim() || '';
      const ensured = await ensureIdeogramOllamaModel(model, endpoint);
      if (!ensured) return;
      provider = ensured as Ideogram4PromptJsonRequest['provider'];
    } else if (['3', 'ideogram', 'ideogram_magic', 'api'].includes(normalizedChoice)) {
      provider = 'ideogram_magic';
      model = '';
      endpoint = '';
    } else if (['4', 'openai', 'openai-compatible', 'public'].includes(normalizedChoice)) {
      provider = 'openai_compatible';
      model = window.prompt('OpenAI-compatible model id:', model || 'gemma-3-4b-it')?.trim() || '';
      if (!model) return;
      endpoint = window.prompt('OpenAI-compatible base URL:', endpoint || 'https://api.openai.com/v1')?.trim() || '';
      if (!endpoint) return;
    } else if (['5', 'template', 'fallback'].includes(normalizedChoice)) {
      provider = 'template';
      model = '';
      endpoint = '';
    } else {
      setLocalError('Magic Prompt provider was not recognized.');
      return;
    }
    setIdeogramPromptProvider(provider);
    setIdeogramPromptModel(model);
    setIdeogramPromptEndpoint(endpoint);
    setIdeogramPromptMessage('');
    await ideogramPromptJson.mutateAsync({
      prompt: generation.prompt,
      width: generation.width,
      height: generation.height,
      regions: generation.promptRegions,
      provider,
      model,
      endpoint,
    });
  }

  async function generate() {
    const error = validateGeneration({
      generation, ideogram4Mode, videoMode, controlNetCompatible,
      payload, controlNetModelCount: controlNetModels.length,
    });
    if (error) {
      setLocalError(error);
      return;
    }
    setLocalError('');
    setJobId('');
    await startMutation.mutateAsync(payload);
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
      <span className="route-sentinel" data-route="studio" aria-label="Studio workspace" />

      <div className="studio-columns">
        <aside className="surface tool-panel studio-controls-panel">
          <button className="collapse-handle" type="button" onClick={() => ui.toggleStudioControls()} title={ui.studioControlsCollapsed ? 'Show controls' : 'Hide controls'}>
            {ui.studioControlsCollapsed ? <PanelLeftOpen size={16} /> : <PanelLeftClose size={16} />}
          </button>
          <button type="button" className="primary-button studio-generate-top" onClick={generate} disabled={processing}>
            {processing ? <LoaderCircle className="spin" size={15} /> : <Play size={15} />}
            Generate
          </button>
          {(localError || activeJob?.status === 'failed' || activeJob?.status === 'completed') && (
            <div className={localError || activeJob?.status === 'failed' ? 'studio-inline-status error' : 'studio-inline-status'}>
              {localError || activeJob?.error || activeJob?.message || generatedOutput?.filename}
            </div>
          )}
          {!ltxDirectorView && !music3Mode && (
            <div className="button-grid mode-switch">
              {qwenImageEdit ? (
                <button className="active" type="button" onClick={() => generation.setActivity('img2img')}>img2img</button>
              ) : (
                <>
                  <button className={generation.activity === 'txt2img' ? 'active' : ''} type="button" onClick={() => generation.setActivity('txt2img')}>txt2img</button>
                  <button className={generation.activity === 'img2img' ? 'active' : ''} type="button" onClick={() => generation.setActivity('img2img')}>img2img</button>
                </>
              )}
            </div>
          )}

          {!music3Mode && <label className="field">
            <span className="field-title-row">
              Prompt
              {ideogram4Mode && (
                <button
                  className="mini-button"
                  type="button"
                  onClick={() => void handleIdeogramMagicPrompt()}
                  disabled={ideogramPromptJson.isPending || !generation.prompt.trim()}
                  title="Convert prompt to Ideogram 4 JSON"
                >
                  {ideogramPromptJson.isPending ? <LoaderCircle className="spin" size={13} /> : <Wand2 size={13} />}
                  Magic JSON
                </button>
              )}
            </span>
            <textarea value={generation.prompt} onChange={(event) => generation.setPrompt(event.currentTarget.value)} placeholder="Describe the image..." />
          </label>}

          {!music3Mode && <label className="field">
            <span>Negative Prompt</span>
            <textarea value={generation.negativePrompt} onChange={(event) => generation.setNegativePrompt(event.currentTarget.value)} placeholder="Avoid..." />
          </label>}

          {music3Mode && (
            <div className="control-stack music3-panel">
              <label className="field"><span>Music description / Caption</span><textarea value={generation.musicCaption} onChange={(event) => generation.setMusicCaption(event.currentTarget.value)} placeholder="Genre, mood, BPM, key, vocals and arrangement..." /></label>
              <label className="field"><span>Generation mode</span><select value={generation.musicMode} onChange={(event) => generation.setMusicMode(event.currentTarget.value as 'instrumental' | 'instrumental_fx' | 'lyrics' | 'caption')}><option value="instrumental">Instrumental — no vocals</option><option value="instrumental_fx">Instrumental + vocal textures — no lyrics</option><option value="lyrics">Lyrics + caption — vocals</option><option value="caption">Caption + negative guidance</option></select></label>
              <label className="field"><span>Lyrics (optional)</span><textarea value={generation.musicMode === 'instrumental' ? '[instrumental]' : generation.musicLyrics} disabled={generation.musicMode !== 'lyrics'} onChange={(event) => generation.setMusicLyrics(event.currentTarget.value)} placeholder="[Intro]\n...\n[Verse]\n...\n[Chorus]\n..." /></label>
              <label className="field"><span>Negative guidance (optional)</span><textarea value={generation.musicNegativePrompt} onChange={(event) => generation.setMusicNegativePrompt(event.currentTarget.value)} placeholder="Avoid instruments, moods, voices or styles..." /></label>
              <div className="two-col">
                <label className="field"><span>Duration (seconds)</span><input type="number" min={0.04} max={300} step={0.04} value={generation.musicDurationSeconds} onChange={(event) => generation.setMusicDurationSeconds(Number(event.currentTarget.value))} /></label>
                <label className="field"><span>Steps</span><input type="number" min={1} max={100} value={generation.steps} onChange={(event) => generation.setSteps(Number(event.currentTarget.value))} /></label>
              </div>
              <div className="two-col">
                <label className="field"><span>CFG scale</span><input type="number" min={0} max={100} step={0.1} value={generation.musicCfgScale} onChange={(event) => generation.setMusicCfgScale(Number(event.currentTarget.value))} /></label>
                <label className="field"><span>Top-K</span><input type="number" min={1} max={4096} value={generation.musicTopK} onChange={(event) => generation.setMusicTopK(Number(event.currentTarget.value))} /></label>
              </div>
              <div className="two-col">
                <label className="field"><span>Output format</span><select value={generation.musicFormat} onChange={(event) => generation.setMusicFormat(event.currentTarget.value as 'flac' | 'mp3' | 'opus')}><option value="mp3">MP3 (V0)</option><option value="flac">FLAC</option><option value="opus">Opus</option></select></label>
                <label className="toggle-row"><input type="checkbox" checked={generation.musicTiledDecode} onChange={(event) => generation.setMusicTiledDecode(event.currentTarget.checked)} /><span>Tiled decode (RTX 3060)</span></label>
              </div>
              <details className="control-section" open><summary>Music 3 Dependencies <span>{music3Status.data?.generation_ready ? 'Ready' : 'Download required'}</span></summary><div className="control-stack"><div className={music3Status.data?.generation_ready ? 'studio-inline-status' : 'studio-inline-status error'}>{music3Status.isLoading ? 'Checking Music 3 assets...' : music3Status.data?.generation_ready ? 'Native Music 3 nodes and models ready.' : `${formatBytes(music3Status.data?.estimated_missing_required_bytes)} required assets missing.`}</div>{music3Status.data?.missing_core_nodes?.length ? <p className="compact-note">ComfyUI core support missing: {music3Status.data.missing_core_nodes.join(', ')}. Run the normal ComfyUI update, then restart when no training job is active.</p> : null}<button type="button" className="primary-button" disabled={music3Download.isPending || !(music3Status.data?.missing_required_assets?.length)} onClick={() => music3Download.mutate((music3Status.data?.missing_required_assets ?? []).map((asset) => asset.key))}>{music3Download.isPending ? 'Downloading...' : 'Download Music 3 base files'}</button><p className="compact-note">RTX 3060 uses INT8 + tiled decode. RTX 5090 can use FP16 and non-tiled decode. Native Music 3 support is delivered by current ComfyUI core; no third-party node is required.</p></div></details>
            </div>
          )}

          {!music3Mode && <div className="img2img-source reference-source-block">
            <label className="dropzone small-dropzone">
              <input
                type="file"
                accept="image/*"
                onChange={async (event) => {
                  const file = event.currentTarget.files?.[0];
                  if (!file) return;
                  generation.setReferenceImage(await readFileAsDataUrl(file), file.name);
                  if (qwenImageEdit || ideogram4Mode || krea2Mode) generation.setActivity('img2img');
                }}
              />
              <span>{generation.referenceImageName || (qwenImageEdit ? 'Select Qwen edit reference image' : ideogram4Mode ? 'Select Ideogram layout reference' : krea2Mode ? 'Select Krea 2 style reference' : 'Select reference image')}</span>
            </label>
            <label className="dropzone small-dropzone multi-reference-dropzone">
              <input
                type="file"
                accept="image/*"
                multiple
                onChange={async (event) => {
                  if (!event.currentTarget.files?.length) return;
                  generation.addExtraReferenceImages(await readFilesAsReferenceImages(event.currentTarget.files));
                  event.currentTarget.value = '';
                }}
              />
              <Images size={16} />
              <span>{generation.extraReferenceImages.length ? `${generation.extraReferenceImages.length} extra reference(s)` : (krea2Mode ? 'Add up to 3 style references' : 'Add multi-image references')}</span>
            </label>
            {generation.extraReferenceImages.length > 0 && (
              <div className="reference-chip-list">
                {generation.extraReferenceImages.map((image, index) => (
                  <button type="button" key={`${image.name}-${index}`} onClick={() => generation.removeExtraReferenceImage(index)} title="Remove reference">
                    <img src={image.dataUrl} alt={image.name} />
                    <span>{index === 0 && generation.preset === 'Wan' ? 'End frame' : image.name}</span>
                    <X size={12} />
                  </button>
                ))}
              </div>
            )}
            {generation.referenceImage && <img src={generation.referenceImage} alt="img2img reference" />}
            {(generation.activity === 'img2img' || viewMode === 'inpaint' || qwenImageEdit) && (
              <>
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
                  <>
                    <label className="field">
                      <span>Engine</span>
                      <select value={generation.inpaintEngine} onChange={(event) => generation.setInpaintEngine(event.currentTarget.value as 'lanpaint' | 'differential' | 'default')}>
                        <option value="lanpaint">LanPaint</option>
                        <option value="differential">Differential Diffusion</option>
                        <option value="default">Default VAE Inpaint</option>
                      </select>
                    </label>
                    <div className="two-col">
                      <label className="field">
                        <span>Mask Blur</span>
                        <input type="number" min={0} max={64} value={generation.maskBlur} onChange={(event) => generation.setMaskBlur(Number(event.currentTarget.value))} />
                      </label>
                      {generation.inpaintEngine === 'lanpaint' ? (
                        <label className="field">
                          <span>Think Steps</span>
                          <input type="number" min={0} max={100} value={generation.lanpaintThinkingSteps} onChange={(event) => generation.setLanpaintThinkingSteps(Number(event.currentTarget.value))} />
                        </label>
                      ) : (
                        <label className="field">
                          <span>Diff Strength</span>
                          <input type="number" min={0} max={1} step={0.01} value={generation.differentialStrength} onChange={(event) => generation.setDifferentialStrength(Number(event.currentTarget.value))} />
                        </label>
                      )}
                    </div>
                  </>
                )}
              </>
            )}
          </div>}

          {ideogram4Mode && (
            <details className="control-section" open>
              <summary>Ideogram 4 Dependencies <span>{ideogram4Status.data?.generation_ready ? 'Ready' : 'Optional'}</span></summary>
              <div className="control-stack ideogram-assets-panel">
                <div className={ideogram4Status.data?.generation_ready ? 'studio-inline-status' : 'studio-inline-status error'}>
                  {ideogram4Status.isLoading
                    ? 'Checking Ideogram 4 assets...'
                    : ideogram4Status.data?.generation_ready
                      ? 'Required Ideogram 4 assets detected.'
                      : (ideogram4Status.data?.missing_core_nodes?.length ?? 0) > 0
                        ? `Comfy runtime needs Ideogram 4 support: ${ideogram4Status.data?.missing_core_nodes?.join(', ')}.`
                        : `${formatBytes(ideogram4Status.data?.estimated_missing_required_bytes)} required assets missing.`}
                </div>
                {ideogram4Status.data?.runtime_checked === false && (
                  <p className="compact-note">Comfy core support is checked after the runtime starts.</p>
                )}
                <div className="asset-check-list">
                  {ideogramAssets.map((asset) => {
                    const checked = selectedIdeogramAssets.includes(asset.key);
                    return (
                      <label key={asset.key} className={asset.installed ? 'asset-check installed' : 'asset-check'}>
                        <input
                          type="checkbox"
                          checked={checked}
                          disabled={asset.installed || ideogram4Download.isPending}
                          onChange={(event) => {
                            setIdeogramAssetSelection((current) => {
                              const base = current.length ? current : (ideogram4Status.data?.missing_required_assets ?? []).map((item) => item.key);
                              return event.currentTarget.checked
                                ? Array.from(new Set([...base, asset.key]))
                                : base.filter((key) => key !== asset.key);
                            });
                          }}
                        />
                        <span>
                          <strong>{asset.label}</strong>
                          <em>{asset.installed ? 'installed' : `${asset.scope || asset.kind} - ${formatBytes(asset.size_bytes_min || asset.size_bytes)}`}</em>
                        </span>
                      </label>
                    );
                  })}
                </div>
                <button
                  type="button"
                  className="primary-button"
                  disabled={ideogram4Download.isPending || !selectedIdeogramAssets.length}
                  onClick={() => ideogram4Download.mutate(selectedIdeogramAssets)}
                >
                  {ideogram4Download.isPending ? <LoaderCircle className="spin" size={14} /> : <FilePlus2 size={14} />}
                  Download selected
                </button>
                <p className="compact-note">The local img2img image is shown as a layout guide for regional JSON; current open Ideogram 4 Comfy route is text-to-image.</p>
              </div>
            </details>
          )}

          {krea2Mode && (
            <details className="control-section" open>
              <summary>Krea 2 Dependencies <span>{krea2Status.data?.generation_ready ? 'Ready' : 'Download required'}</span></summary>
              <div className="control-stack ideogram-assets-panel">
                <div className={krea2Status.data?.generation_ready ? 'studio-inline-status' : 'studio-inline-status error'}>
                  {krea2Status.isLoading
                    ? 'Checking Krea 2 assets...'
                    : krea2Status.data?.generation_ready
                      ? 'Krea 2 local model and native Comfy nodes are ready.'
                      : `${formatBytes(krea2Status.data?.estimated_missing_required_bytes)} required assets missing.`}
                </div>
                {krea2Status.data?.runtime_checked === false && <p className="compact-note">Core support is checked after the runtime starts. No backend restart was requested.</p>}
                <div className="asset-check-list">
                  {(krea2Status.data?.assets ?? []).map((asset) => (
                    <label key={asset.key} className={asset.installed ? 'asset-check installed' : 'asset-check'}>
                      <input type="checkbox" checked={!asset.installed} disabled={asset.installed || krea2Download.isPending} readOnly />
                      <span><strong>{asset.label}</strong><em>{asset.installed ? 'installed' : `${asset.scope || 'dependency'} - ${formatBytes(asset.size_bytes || asset.size_bytes_min)}`}</em></span>
                    </label>
                  ))}
                </div>
                <button
                  type="button"
                  className="primary-button"
                  disabled={krea2Download.isPending || !(krea2Status.data?.missing_required_assets?.length)}
                  onClick={() => krea2Download.mutate((krea2Status.data?.missing_required_assets ?? []).map((asset) => asset.key))}
                >
                  {krea2Download.isPending ? <LoaderCircle className="spin" size={14} /> : <FilePlus2 size={14} />}
                  Download Krea 2 base files
                </button>
                {!!krea2Status.data?.missing_optional_assets?.some((asset) => asset.kind === 'lora') && (
                  <button
                    type="button"
                    className="secondary-button"
                    disabled={krea2Download.isPending}
                    onClick={() => krea2Download.mutate((krea2Status.data?.missing_optional_assets ?? []).filter((asset) => asset.kind === 'lora').map((asset) => asset.key))}
                  >
                    {krea2Download.isPending ? <LoaderCircle className="spin" size={14} /> : <FilePlus2 size={14} />}
                    Download optional Krea 2 LoRAs
                  </button>
                )}
                <p className="compact-note">Detected profile: {krea2Status.data?.recommended_profile === 'rtx_5090' ? 'RTX 5090 / 1024px' : 'shared VRAM / 768px'} ({krea2Status.data?.detected_vram_gb?.toFixed(1) || 'unknown'} GB VRAM).</p>
                <p className="compact-note">Official local route: text-to-image or up to 3 ordered reference images. Optional `krea2_style_reference.safetensors` enables the style adapter; ordinary Krea LoRAs remain selectable from Concepts.</p>
              </div>
            </details>
          )}

          {ideogram4Mode && (
            <details className="control-section" open>
              <summary>Magic Prompt JSON <span>{ideogramPromptProvider === 'comfy_gemma4' ? 'Comfy Gemma4' : ideogramPromptProvider === 'ollama' ? 'Gemma' : ideogramPromptProvider}</span></summary>
              <div className="control-stack ideogram-magic-panel">
                <label className="field">
                  <span>Provider</span>
                  <select value={ideogramPromptProvider} onChange={(event) => setIdeogramPromptProvider(event.currentTarget.value as typeof ideogramPromptProvider)}>
                    <option value="comfy_gemma4">Comfy Gemma4 local</option>
                    <option value="ollama">Ollama / Gemma local</option>
                    <option value="ideogram_magic">Ideogram Magic Prompt API</option>
                    <option value="openai_compatible">OpenAI-compatible endpoint</option>
                    <option value="template">Local template fallback</option>
                  </select>
                </label>
                {(ideogramPromptProvider === 'ollama' || ideogramPromptProvider === 'openai_compatible') && (
                  <label className="field">
                    <span>Model</span>
                    <input value={ideogramPromptModel} onChange={(event) => setIdeogramPromptModel(event.currentTarget.value)} placeholder={ideogramPromptProvider === 'ollama' ? 'gemma4:e2b, gemma3:4b or gemma3:1b' : 'model id'} />
                  </label>
                )}
                {(ideogramPromptProvider === 'ollama' || ideogramPromptProvider === 'openai_compatible') && (
                  <label className="field">
                    <span>Endpoint</span>
                    <input value={ideogramPromptEndpoint} onChange={(event) => setIdeogramPromptEndpoint(event.currentTarget.value)} placeholder={ideogramPromptProvider === 'ollama' ? 'http://127.0.0.1:11434' : 'https://.../v1'} />
                  </label>
                )}
                <button
                  type="button"
                  className="primary-button"
                  onClick={() => void handleIdeogramMagicPrompt()}
                  disabled={ideogramPromptJson.isPending || !generation.prompt.trim()}
                >
                  {ideogramPromptJson.isPending ? <LoaderCircle className="spin" size={14} /> : <Wand2 size={14} />}
                  Convert prompt to JSON
                </button>
                {ideogramPromptMessage && <div className="studio-inline-status">{ideogramPromptMessage}</div>}
                <p className="compact-note">Short prompts and pasted JSON both stay valid: JSON is normalized; plain text is expanded into the native Ideogram 4 caption schema.</p>
              </div>
            </details>
          )}

          <details className="control-section" open>
            <summary>Concepts (LoRA) <button className="mini-button" type="button" onClick={(event) => { event.preventDefault(); setLoraModalOpen(true); }}><FilePlus2 size={13} /> Add</button></summary>
            <div className="control-stack">
              {loraStore.activeLoras.length > 0 ? (
                <div className="active-lora-list">
                  {loraStore.activeLoras.map((lora) => (
                    <div className="active-lora" key={lora.relative_name}>
                      <span title={lora.relative_name}>{lora.name}</span>
                      <input type="number" min={-2} max={2} step={0.05} value={lora.strength} onChange={(event) => loraStore.updateStrength(lora.relative_name, Number(event.currentTarget.value))} />
                      <button type="button" onClick={() => loraStore.removeLora(lora.relative_name)}>Remove</button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="compact-note">No active LoRA concepts.</p>
              )}
            </div>
          </details>

          <details className="control-section">
            <summary>VAE / Text Encoder</summary>
            <div className="control-stack">
              <div className="control-row">
                <span>Text encoder override</span>
                <button className={generation.textEncoderOverrideEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setTextEncoderOverrideEnabled(!generation.textEncoderOverrideEnabled)} aria-label="Toggle text encoder override">
                  <span />
                </button>
              </div>
              {generation.textEncoderOverrideEnabled && (
                <label className="field">
                  <span>Text Encoder</span>
                  <select value={generation.textEncoder} onChange={(event) => generation.setTextEncoder(event.currentTarget.value)}>
                    <option value="Automatic">Automatic</option>
                    {textEncoderOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
              )}
              <div className="control-row">
                <span>VAE override</span>
                <button className={generation.vaeOverrideEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setVaeOverrideEnabled(!generation.vaeOverrideEnabled)} aria-label="Toggle VAE override">
                  <span />
                </button>
              </div>
              {generation.vaeOverrideEnabled && (
                <label className="field">
                  <span>VAE / Video VAE</span>
                  <select value={generation.videoVae} onChange={(event) => generation.setVideoVae(event.currentTarget.value)}>
                    <option value="Automatic">Automatic</option>
                    {vaeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
              )}
            </div>
          </details>

          <details className="control-section">
            <summary>{generation.preset === 'Qwen' ? 'Lightning LoRA' : 'Distilled LoRA'}</summary>
            <div className="control-stack">
              {generation.preset === 'Qwen' ? (
                <div className="control-row">
                  <span>Auto Qwen edit lightning LoRA</span>
                  <button className={generation.lightningLoraEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setLightningLoraEnabled(!generation.lightningLoraEnabled)} aria-label="Toggle Qwen lightning LoRA">
                    <span />
                  </button>
                </div>
              ) : (
                <div className="control-row">
                  <span>Auto distilled route LoRAs</span>
                  <button className={generation.distilledLoraEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setDistilledLoraEnabled(!generation.distilledLoraEnabled)} aria-label="Toggle distilled LoRAs">
                    <span />
                  </button>
                </div>
              )}
              <p className="compact-note">{generation.preset === 'Qwen' ? 'Sends qwen_auto_edit_lora directly to backend.' : 'When enabled, backend resolves the default distilled LoRA stack for the selected route.'}</p>
            </div>
          </details>

          <details className="control-section" open>
            <summary>Frame Dimensions</summary>
            <div className="control-stack">
              <div className="two-col">
                <label className="field"><span>Width</span><input type="number" min={64} step={8} value={generation.width} onChange={(event) => generation.setSize(Number(event.currentTarget.value), generation.height)} /></label>
                <label className="field"><span>Height</span><input type="number" min={64} step={8} value={generation.height} onChange={(event) => generation.setSize(generation.width, Number(event.currentTarget.value))} /></label>
              </div>
              <div className="button-grid">
                <button type="button" onClick={() => generation.setSize(832, 480)}>16:9</button>
                <button type="button" onClick={() => generation.setSize(512, 512)}>1:1</button>
              </div>
            </div>
          </details>

          <details className="control-section" open>
            <summary>Generation Settings <span>Synced</span></summary>
            <div className="control-stack">
              <div className="two-col">
                <label className="field"><span>Sampling Method</span><select value={generation.sampler} onChange={(event) => generation.setSampler(event.currentTarget.value)}><option value="euler_ancestral">Euler Ancestral</option><option value="euler_ancestral_cfg_pp">Euler CFG++</option><option value="euler">Euler</option><option value="dpmpp_2m">DPM++ 2M</option><option value="dpmpp_sde">DPM++ SDE</option></select></label>
                <label className="field"><span>Schedule Type</span><select value={generation.scheduler} onChange={(event) => generation.setScheduler(event.currentTarget.value)}><option value="karras">Karras</option><option value="normal">Normal</option><option value="quadratic">Quadratic</option><option value="simple">Simple</option><option value="sgm_uniform">SGM Uniform</option></select></label>
              </div>
              <div className="two-col">
                <label className="field"><span>Steps</span><input type="number" min={1} max={150} value={generation.steps} onChange={(event) => generation.setSteps(Number(event.currentTarget.value))} /></label>
                <label className="field"><span>CFG Scale</span><input type="number" min={0} max={30} step={0.1} value={generation.cfg} onChange={(event) => generation.setCfg(Number(event.currentTarget.value))} /></label>
              </div>
              <label className="field"><span>Seed</span><input type="number" value={generation.seed} onChange={(event) => generation.setSeed(Number(event.currentTarget.value))} /></label>
            </div>
          </details>

          {videoMode && (
            <details className="control-section" open={generation.preset.toLowerCase() === 'ltx'}>
              <summary>Video / Motion</summary>
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
              <div className="control-row">
                <span>Video VAE override</span>
                <button className={generation.vaeOverrideEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setVaeOverrideEnabled(!generation.vaeOverrideEnabled)} aria-label="Toggle video VAE override">
                  <span />
                </button>
              </div>
              {generation.vaeOverrideEnabled && (
                <label className="field">
                  <span>Video VAE</span>
                  <select value={generation.videoVae} onChange={(event) => generation.setVideoVae(event.currentTarget.value)}>
                    <option value="Automatic">Automatic</option>
                    {vaeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
              )}
              <div className="control-row">
                <span>Audio VAE route</span>
                <button className={generation.audioVaeEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setAudioVaeEnabled(!generation.audioVaeEnabled)} aria-label="Toggle audio VAE route">
                  <span />
                </button>
              </div>
              {generation.audioVaeEnabled && (
                <label className="field">
                  <span>Audio VAE</span>
                  <select value={generation.audioVae} onChange={(event) => generation.setAudioVae(event.currentTarget.value)}>
                    <option value="Automatic">Automatic</option>
                    {audioVaeOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                  </select>
                </label>
              )}
              {generation.preset.toLowerCase() === 'ltx' && (
                <>
                  <div className="control-row">
                    <span>Active audio</span>
                    <button className={generation.videoActiveAudio ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setVideoActiveAudio(!generation.videoActiveAudio)} aria-label="Toggle active audio">
                      <span />
                    </button>
                  </div>
                  <div className="control-row">
                    <span>Latent upscale route</span>
                    <button className={generation.latentUpscaleEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setLatentUpscaleEnabled(!generation.latentUpscaleEnabled)} aria-label="Toggle latent upscale route">
                      <span />
                    </button>
                  </div>
                  {generation.latentUpscaleEnabled && (
                    <label className="field">
                      <span>Latent Upscale</span>
                      <select value={generation.latentUpscale} onChange={(event) => generation.setLatentUpscale(event.currentTarget.value)}>
                        <option value="Automatic">Automatic</option>
                        {latentUpscaleOptions.map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                      </select>
                    </label>
                  )}
                  <div className="control-row">
                    <span>Latent upscale refine</span>
                    <button className={generation.latentUpscaleRefine ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setLatentUpscaleRefine(!generation.latentUpscaleRefine)} disabled={!generation.latentUpscaleEnabled} aria-label="Toggle latent upscale refine">
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
            </details>
          )}

          <details className="control-section">
            <summary>ControlNet / Reference</summary>
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
                        <option value="dwpose">DWPose</option>
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
          </details>

          {generation.preset.toLowerCase() === 'ltx' && (
            <details className="control-section" open>
              <summary>LTX 2.3 Director</summary>
            <section className="director-panel">
              <div className="control-row">
                <span>LTX Director</span>
                <button className={generation.directorEnabled ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setDirectorEnabled(!generation.directorEnabled)} aria-label="Toggle LTX Director">
                  <span />
                </button>
              </div>
              {generation.directorEnabled && (
                <div className="control-stack">
                  <div className="two-col">
                    <label className="field">
                      <span>Duration</span>
                      <input type="number" min={0.5} step={0.25} value={generation.directorDuration} onChange={(event) => generation.setDirectorTiming(Number(event.currentTarget.value), generation.directorGuideStrength)} />
                    </label>
                    <label className="field">
                      <span>Guide {generation.directorGuideStrength.toFixed(2)}</span>
                      <input type="range" min={0} max={2} step={0.05} value={generation.directorGuideStrength} onChange={(event) => generation.setDirectorTiming(generation.directorDuration, Number(event.currentTarget.value))} />
                    </label>
                  </div>
                  <label className="field">
                    <span>Local Prompt</span>
                    <textarea value={generation.directorLocalPrompt} onChange={(event) => generation.setDirectorPrompts(event.currentTarget.value, generation.directorLocalNegative)} placeholder="Segment prompt override..." />
                  </label>
                  <label className="field">
                    <span>Local Negative</span>
                    <textarea value={generation.directorLocalNegative} onChange={(event) => generation.setDirectorPrompts(generation.directorLocalPrompt, event.currentTarget.value)} placeholder="Segment negative override..." />
                  </label>
                  <div className="three-col">
                    <label className="field">
                      <span>Resize</span>
                      <select value={generation.directorResizeMethod} onChange={(event) => generation.setDirectorResize(event.currentTarget.value, generation.directorDivisibleBy, generation.directorImgCompression)}>
                        <option value="maintain aspect ratio">Maintain aspect</option>
                        <option value="crop">Crop</option>
                        <option value="stretch">Stretch</option>
                        <option value="pad">Pad</option>
                      </select>
                    </label>
                    <label className="field">
                      <span>Divisible</span>
                      <input type="number" min={1} max={256} value={generation.directorDivisibleBy} onChange={(event) => generation.setDirectorResize(generation.directorResizeMethod, Number(event.currentTarget.value), generation.directorImgCompression)} />
                    </label>
                    <label className="field">
                      <span>Img CRF</span>
                      <input type="number" min={0} max={100} value={generation.directorImgCompression} onChange={(event) => generation.setDirectorResize(generation.directorResizeMethod, generation.directorDivisibleBy, Number(event.currentTarget.value))} />
                    </label>
                  </div>
                  <div className="control-row">
                    <span>Custom audio</span>
                    <button className={generation.directorUseCustomAudio ? 'toggle active' : 'toggle'} type="button" onClick={() => generation.setDirectorUseCustomAudio(!generation.directorUseCustomAudio)} aria-label="Toggle Director custom audio">
                      <span />
                    </button>
                  </div>
                  <p className="compact-note">Director sends timeline metadata to LTX workflows with LTXDirector nodes. Multi-image references are included in order.</p>
                </div>
              )}
            </section>
            </details>
          )}

          <details className="control-section">
            <summary>Refiner</summary>
            <p className="compact-note">Refiner controls are template-aware and remain disabled unless the selected workflow exposes them.</p>
          </details>
          <details className="control-section">
            <summary>Advanced</summary>
            <p className="compact-note">{generation.modelName || generation.modelPath ? `Checkpoint: ${generation.modelName || generation.modelPath}` : 'Checkpoint: Automatic'} / Workflow: {generation.workflowName || 'Default backend route'}</p>
          </details>
        </aside>

        <main className="surface preview-panel studio-preview">
          <div className="studio-subheader">
            <div />
            <div className="studio-view-strip">
            {generation.preset.toLowerCase() === 'ltx' && <button className={viewMode === 'director' ? 'active' : ''} type="button" onClick={() => { setViewMode('director'); generation.setDirectorEnabled(true); }}><Clapperboard size={13} /> LTX 2.3 Director</button>}
            <button className={viewMode === 'linear' ? 'active' : ''} type="button" onClick={() => setViewMode('linear')}>Linear Viewer</button>
            {!music3Mode && <button
              className={viewMode === 'inpaint' ? 'active' : ''}
              type="button"
              onClick={() => {
                setViewMode('inpaint');
                generation.setActivity('img2img');
                generation.setImg2ImgMode('inpaint');
              }}
            >
              Inpaint Canvas
            </button>}
            <button className={viewMode === 'workflow' ? 'active' : ''} type="button" onClick={() => setViewMode('workflow')}>
              <Workflow size={13} />
              Node Workflow
            </button>
            </div>
            <button className="studio-gallery-toggle" type="button" onClick={() => ui.toggleStudioGallery()}><Images size={14} /> Gallery</button>
          </div>
          <div className="preview-actions">
            <button type="button" className="icon-button" title="Clear preview" onClick={() => setJobId('')}><X size={14} /></button>
            <button type="button" className="icon-button" title="Adjust preview"><SlidersHorizontal size={14} /></button>
            <button type="button" className="icon-button" title="Save preview"><Save size={14} /></button>
          </div>
          {viewMode === 'linear' && (
            <div className="studio-preview-content linear-viewer" onWheel={handleLinearViewerWheel}>
              <div className="linear-zoom-stage" style={{ transform: `scale(${linearZoom})` }}>
                {ideogram4Mode ? (
                  <IdeogramRegionEditor generation={generation} sourceUrl={generation.activity === 'img2img' ? generation.referenceImage || previewUrl : previewUrl} previewIsVideo={previewIsVideo && generation.activity !== 'img2img'} />
                ) : previewUrl ? (
                  previewIsVideo ? <video className="extras-media" src={previewUrl} controls playsInline /> : previewIsAudio ? <audio className="music3-audio-player" src={previewUrl} controls /> : <img className="extras-media" src={previewUrl} alt="Studio output preview" />
                ) : (
                  <div className="preview-empty"><Images size={38} /><p>No output loaded</p><span>Generated files appear from ./output</span></div>
                )}
              </div>
              <div className="linear-zoom-readout">
                <button type="button" onClick={() => setLinearZoom((value) => Math.max(0.25, Number((value - 0.1).toFixed(2))))}>-</button>
                <span>{Math.round(linearZoom * 100)}%</span>
                <button type="button" onClick={() => setLinearZoom((value) => Math.min(3, Number((value + 0.1).toFixed(2))))}>+</button>
              </div>
            </div>
          )}
          {viewMode === 'inpaint' && (
            <div className="studio-inpaint-workspace">
              <div className="inpaint-tool-rail"><button className="active" type="button"><Brush size={16} /></button><button type="button"><Grid3X3 size={16} /></button><button type="button"><X size={16} /></button></div>
              <div className="studio-inpaint-stage">
                {generation.referenceImage ? <InpaintCanvas image={generation.referenceImage} brushSize={generation.brushSize} onBrushSizeChange={generation.setBrushSize} onMaskChange={generation.setInpaintMaskImage} /> : <div className="preview-empty"><span>No source image</span></div>}
              </div>
              <div className="inpaint-bottom-bar"><span>Brush</span><input type="range" min={4} max={128} value={generation.brushSize} onChange={(event) => generation.setBrushSize(Number(event.currentTarget.value))} /><strong>{generation.brushSize}px</strong></div>
            </div>
          )}
          {viewMode === 'workflow' && (
            <div className="studio-node-workspace">
              <StudioNodeCapabilityStatus
                report={nodeCapabilities.data}
                count={nodeCapabilities.data?.required_count ?? workflowClassCount}
                loading={nodeCapabilities.isLoading && workflowClassCount > 0}
                error={nodeCapabilities.isError}
              />
              {studioWorkflow && <StudioWorkflowInputStatus
                report={workflowInputReport.data}
                loading={workflowInputReport.isLoading}
                error={workflowInputReport.isError}
              />}
              {workflowAnalysis.isLoading ? (
                <div className="preview-empty"><LoaderCircle className="spin" size={30} /><p>Loading backend workflow graph</p></div>
              ) : studioWorkflowGraph.nodes.length ? (
                <StudioWorkflowGraph
                  nodes={studioWorkflowGraph.nodes}
                  links={studioWorkflowGraph.links}
                  width={studioWorkflowGraph.width}
                  height={studioWorkflowGraph.height}
                  workflowName={studioWorkflow?.name || generation.workflowName || 'Default backend route'}
                  synced={Boolean(workflowAnalysis.data?.visual_graph?.nodes?.length)}
                />
              ) : (
                <div className="preview-empty"><Workflow size={36} /><p>No workflow graph loaded</p><span>Select or import a workflow to sync nodes.</span></div>
              )}
            </div>
          )}
          {viewMode === 'director' && (
            <div className="director-suite">
              <header className="director-engine-bar">
                <div className="director-port-legend">
                  <span><i className="port model" />model</span>
                  <span><i className="port clip" />clip</span>
                  <span><i className="port audio" />audio_vae</span>
                  <span><i className="port latent" />optional_latent</span>
                </div>
                <strong>LTX Director Engine Map</strong>
                <div className="director-port-legend right">
                  <span>model<i className="port model" /></span>
                  <span>positive<i className="port positive" /></span>
                  <span>video_latent<i className="port video" /></span>
                  <span>combined_audio<i className="port audio" /></span>
                </div>
              </header>
              <div className="director-main">
                <section className="director-workbench">
                  <div className="director-stats">
                    <label>
                      <span>Duration</span>
                      <input type="number" min={0.5} step={0.25} value={generation.directorDuration} onChange={(event) => generation.setDirectorTiming(Number(event.currentTarget.value), generation.directorGuideStrength)} />
                    </label>
                    <label>
                      <span>Frame Rate</span>
                      <input type="number" min={1} max={60} value={generation.videoFps} onChange={(event) => generation.setVideoTiming(generation.videoFrames, Number(event.currentTarget.value), generation.videoSeconds)} />
                    </label>
                    <label>
                      <span>Custom Width</span>
                      <input type="number" min={64} step={8} value={generation.width} onChange={(event) => generation.setSize(Number(event.currentTarget.value), generation.height)} />
                    </label>
                    <label>
                      <span>Custom Height</span>
                      <input type="number" min={64} step={8} value={generation.height} onChange={(event) => generation.setSize(generation.width, Number(event.currentTarget.value))} />
                    </label>
                    <label>
                      <span>Resize Method</span>
                      <select value={generation.directorResizeMethod} onChange={(event) => generation.setDirectorResize(event.currentTarget.value, generation.directorDivisibleBy, generation.directorImgCompression)}>
                        <option value="maintain aspect ratio">Maintain aspect ratio</option>
                        <option value="crop">Crop</option>
                        <option value="stretch">Stretch</option>
                        <option value="pad">Pad</option>
                      </select>
                    </label>
                  </div>
                  <div className="director-toolbar">
                    <label className="director-tool-button">
                      <input
                        type="file"
                        accept="image/*"
                        onChange={async (event) => {
                          const file = event.currentTarget.files?.[0];
                          if (!file) return;
                          generation.setReferenceImage(await readFileAsDataUrl(file), file.name);
                          generation.setActivity('img2img');
                          generation.setImg2ImgMode('image');
                        }}
                      />
                      <FilePlus2 size={13} /> Add Image
                    </label>
                    <button type="button" onClick={() => generation.setDirectorPrompts(generation.directorLocalPrompt || 'new text-only motion beat', generation.directorLocalNegative)}>Add Text</button>
                    <button type="button" onClick={() => generation.setDirectorUseCustomAudio(true)}>Add Audio</button>
                    <button type="button" onClick={() => generation.setReferenceImage(null)}>Delete</button>
                    <button type="button" className={allReferenceImages.length > 1 ? 'active' : ''}>Multiimage Load <b>{allReferenceImages.length > 1 ? 'On' : 'Off'}</b></button>
                    <button type="button" className={generation.directorUseCustomAudio ? 'active align-right' : 'align-right'} onClick={() => generation.setDirectorUseCustomAudio(!generation.directorUseCustomAudio)}>Custom Audio: {generation.directorUseCustomAudio ? 'On' : 'Off'}</button>
                  </div>
                  <div className="director-ruler">
                    {[0, 1.5, 3, 4.5, 6, 7.5, 9, 10.5].map((mark) => <span key={mark}>{mark.toFixed(2)}s</span>)}
                  </div>
                  <div className="director-timeline">
                    <div className="director-track video-track">
                      <span className="director-track-label">Video Keyframes Track</span>
                      <article className="director-segment selected">
                        {generation.referenceImage && <img src={generation.referenceImage} alt="Director reference" />}
                        <button type="button" onClick={() => generation.setReferenceImage(null)}>x</button>
                        <strong>Segment #1</strong>
                        <small>0.00s - {Math.min(generation.directorDuration, 3.13).toFixed(2)}s</small>
                        <em>image</em>
                      </article>
                      <article className="director-segment motion">
                        <strong>Segment #2</strong>
                        <small>{Math.min(generation.directorDuration, 3.14).toFixed(2)}s - {generation.directorDuration.toFixed(2)}s</small>
                        <em>{generation.directorLocalPrompt || 'new text-only motion beat'}</em>
                      </article>
                      <button type="button" className="director-add-marker" onClick={() => generation.setDirectorPrompts(generation.directorLocalPrompt || 'new text-only motion beat', generation.directorLocalNegative)}>+</button>
                    </div>
                    <div className="director-track audio-track">
                      <span className="director-track-label">Audio Waveform Track</span>
                      <div className="director-audio">{generation.directorUseCustomAudio ? 'custom-audio.mp3' : 'custom audio waveform'}</div>
                      <button type="button" className="director-add-marker" onClick={() => generation.setDirectorUseCustomAudio(true)}>+</button>
                    </div>
                  </div>
                  <div className="director-playbar">
                    <button type="button" className="primary-icon" onClick={() => generation.setDirectorEnabled(true)}><Play size={14} /></button>
                    <button type="button" className="mini-button">Undo</button>
                    <span><b>0.00s</b> Start: 0.00 | End: {Math.min(generation.directorDuration, 3.13).toFixed(2)}</span>
                    <label>
                      Guide Strength
                      <input type="range" min={0} max={2} step={0.05} value={generation.directorGuideStrength} onChange={(event) => generation.setDirectorTiming(generation.directorDuration, Number(event.currentTarget.value))} />
                      <strong>{generation.directorGuideStrength.toFixed(2)}</strong>
                    </label>
                  </div>
                  <div className="director-prompts">
                    <label>
                      <span>Prompt for selected segment</span>
                      <textarea value={generation.directorLocalPrompt} onChange={(event) => generation.setDirectorPrompts(event.currentTarget.value, generation.directorLocalNegative)} placeholder="image" />
                    </label>
                    <label>
                      <span>Negative prompt for selected segment</span>
                      <textarea value={generation.directorLocalNegative} onChange={(event) => generation.setDirectorPrompts(generation.directorLocalPrompt, event.currentTarget.value)} placeholder="Avoid artifacts, blur, noise, bad anatomy, black frames..." />
                    </label>
                  </div>
                </section>
                <aside className="director-inspector">
                  <div className="director-inspector-tabs">
                    <button className="active" type="button">Live Director Preview</button>
                    <button type="button">Info</button>
                  </div>
                  <section className="director-inspector-card">
                    <header><strong>Image Segment</strong><span>Segment #1</span></header>
                    <div className="two-col">
                      <label className="field"><span>Start</span><input type="number" min={0} step={0.01} value={0} readOnly /></label>
                      <label className="field"><span>Duration</span><input type="number" min={0.25} step={0.01} value={Math.min(generation.directorDuration, 3.13).toFixed(2)} onChange={(event) => generation.setDirectorTiming(Math.max(Number(event.currentTarget.value), generation.directorDuration), generation.directorGuideStrength)} /></label>
                    </div>
                    <div className="two-col">
                      <label className="field"><span>Guide Strength</span><input type="number" min={0} max={2} step={0.05} value={generation.directorGuideStrength} onChange={(event) => generation.setDirectorTiming(generation.directorDuration, Number(event.currentTarget.value))} /></label>
                      <label className="field"><span>Insert Frame</span><input type="number" min={1} value={directorFrames} onChange={(event) => generation.setVideoTiming(Number(event.currentTarget.value), generation.videoFps, generation.videoSeconds)} /></label>
                    </div>
                    <label className="field"><span>Negative prompt for segment</span><textarea value={generation.directorLocalNegative} onChange={(event) => generation.setDirectorPrompts(generation.directorLocalPrompt, event.currentTarget.value)} placeholder="Segment-specific negatives..." /></label>
                  </section>
                  <section className="director-inspector-card">
                    <header><strong>Crop Mode / Camera Keyframes</strong><span>normalized</span></header>
                    <label className="field">
                      <span>Crop Mode</span>
                      <select value={generation.directorResizeMethod} onChange={(event) => generation.setDirectorResize(event.currentTarget.value, generation.directorDivisibleBy, generation.directorImgCompression)}>
                        <option value="maintain aspect ratio">Maintain aspect ratio</option>
                        <option value="crop">Crop visible frame</option>
                        <option value="pad">Pad frame</option>
                        <option value="stretch">Stretch frame</option>
                      </select>
                    </label>
                    <div className="director-crop-preview">
                      {previewUrl ? (
                        previewIsVideo ? <video src={previewUrl} muted playsInline /> : <img src={previewUrl} alt="Director crop preview" />
                      ) : generation.referenceImage ? (
                        <img src={generation.referenceImage} alt="Director reference crop" />
                      ) : (
                        <span>No preview loaded</span>
                      )}
                    </div>
                    <p>Drag the red box to frame the visible crop; resize handles sync the camera crop.</p>
                  </section>
                  <button type="button" className="primary-button director-render-button" onClick={generate} disabled={processing}>
                    {processing ? <LoaderCircle className="spin" size={14} /> : <Play size={14} />}
                    Render Sequenced Video
                  </button>
                  <div className="director-export-row">
                    <button type="button" onClick={() => generation.setDirectorEnabled(true)}>Export Metadata</button>
                    <button type="button" onClick={() => { generation.setActivity('img2img'); generation.setImg2ImgMode('image'); setViewMode('linear'); }}>Send to img2video</button>
                  </div>
                </aside>
              </div>
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

      {loraModalOpen && (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Select LoRA concept">
          <div className="modal-panel lora-modal">
            <header className="modal-header">
              <strong>Select Concept (LoRA)</strong>
              <button className="icon-button" type="button" onClick={() => setLoraModalOpen(false)} title="Close LoRA selector">
                <X size={15} />
              </button>
            </header>
            <div className="modal-search-row">
              <input value={loraSearch} onChange={(event) => setLoraSearch(event.currentTarget.value)} placeholder="Filter by name or tag..." />
              {['All', 'Anima', 'Flux', 'LTX', 'Qwen', 'SDXL'].map((tag) => <button className="mini-button" type="button" key={tag}>{tag}</button>)}
            </div>
            <div className="lora-card-grid">
              {visibleLoras.map((lora) => {
                const relativeName = (lora.relative_path || lora.name).replace(/^loras[\\/]/i, '').replaceAll('/', '\\');
                const active = loraStore.activeLoras.some((item) => item.relative_name === relativeName);
                return (
                  <button
                    key={lora.relative_path || lora.path}
                    className={active ? 'lora-card active' : 'lora-card'}
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
                    <span>{lora.tags?.[0] || generation.preset}</span>
                    <strong>{lora.name}</strong>
                  </button>
                );
              })}
            </div>
            <footer className="modal-footer">
              <span>{loraStore.activeLoras.length} selected concept(s)</span>
              <button className="primary-button" type="button" onClick={() => setLoraModalOpen(false)}>Confirm Selection</button>
            </footer>
          </div>
        </div>
      )}
    </section>
  );
}
