import type { WorkflowGraphLink, WorkflowGraphNode, WorkflowSummary } from '../../api/types';
import type { GenerationState } from '../../stores/generationStore';

// Pure graph projection + scoped preview. No edits are persisted to the backend here.
function workflowNodeKind(node: WorkflowGraphNode) {
  const text = `${node.class_type} ${node.title || ''}`.toLowerCase();
  if (text.includes('director') || text.includes('timeline')) return 'director';
  if (text.includes('checkpoint') || text.includes('unet') || text.includes('model') || text.includes('loader')) return 'model';
  if (text.includes('prompt') || text.includes('text') || text.includes('clip')) return 'text';
  if (text.includes('sampler') || text.includes('scheduler') || text.includes('noise') || text.includes('latent')) return 'sample';
  if (text.includes('vae') || text.includes('image') || text.includes('video') || text.includes('audio')) return 'media';
  if (text.includes('lora') || text.includes('guide') || text.includes('control')) return 'control';
  return 'utility';
}

function shortWorkflowValue(value: unknown) {
  const text = String(value ?? '');
  return text.length > 42 ? `${text.slice(0, 39)}...` : text;
}

function patchedWorkflowNode(
  node: WorkflowGraphNode,
  generation: GenerationState,
  alignedFrames: number,
  directorFrames: number,
): WorkflowGraphNode {
  const title = `${node.title || ''} ${node.class_type}`.toLowerCase();
  const widgets = (node.widgets ?? []).map((widget) => {
    const name = String(widget.name || '').toLowerCase();
    let value = widget.value;
    if (name === 'ckpt_name' || name === 'unet_name') value = generation.modelPath || generation.modelName || value;
    if (name === 'text' && title.includes('positive')) value = generation.prompt || value;
    if (name === 'text' && title.includes('negative')) value = generation.negativePrompt || value;
    if (name === 'width' || name === 'custom_width') value = generation.width;
    if (name === 'height' || name === 'custom_height') value = generation.height;
    if (name === 'length' || name.includes('frame')) value = title.includes('director') ? directorFrames : alignedFrames;
    if (name === 'fps' || name === 'frame_rate') value = generation.videoFps;
    if (name === 'steps') value = generation.preset.toLowerCase() === 'ltx' ? Math.max(8, generation.steps) : generation.steps;
    if (name === 'cfg') value = generation.cfg;
    if (name === 'noise_seed') value = generation.seed;
    if (name === 'sampler_name') value = generation.sampler;
    if (name === 'strength' || name === 'guide_strength') value = generation.directorGuideStrength;
    return { ...widget, value };
  });
  return { ...node, widgets };
}

export function preferredWorkflow(workflows: WorkflowSummary[] | undefined, preset: string, activeId: string) {
  if (!workflows?.length) return undefined;
  const active = workflows.find((workflow) => workflow.id === activeId);
  if (active) return active;
  const key = preset.toLowerCase();
  const tokens: Record<string, string[]> = {
    ltx: ['ltx23-img2vid-512-base', 'ltx'],
    wan: ['wan'],
    qwen: ['qwen-img2img-base', 'qwen'],
    ideogram4: ['ideogram4-kj-prompt-builder', 'ideogram4', 'ideogram'],
    ideogram: ['ideogram4-kj-prompt-builder', 'ideogram4', 'ideogram'],
    anima: ['anima-base', 'anima'],
    zimageturbo: ['zimage-turbo-base', 'zimage'],
    flux: ['flux'],
    music3: ['music3', 'music-3', 'minimax-music'],
    xl: ['sdxl-base', 'sdxl'],
    sdxl: ['sdxl-base', 'sdxl'],
    sd: ['sd15-base', 'sd15'],
  };
  const wanted = tokens[key] || [key];
  return workflows.find((workflow) => wanted.some((token) => workflow.id.toLowerCase().includes(token) || workflow.name.toLowerCase().includes(token) || workflow.tags?.includes(token))) || workflows[0];
}

export function music3WorkflowGraph(generation: GenerationState): { nodes: WorkflowGraphNode[]; links: WorkflowGraphLink[]; width: number; height: number } {
  const nodes: WorkflowGraphNode[] = [
    ['1', 'UNETLoader', 'DiT model'], ['2', 'CLIPLoader', 'Music text encoder'], ['3', 'VAELoader', 'Audio VAE'], ['4', 'MiniMaxMusic3TextEncode', 'Caption + lyrics'], ['5', 'ConditioningZeroOut', 'Negative conditioning'], ['6', 'EmptyMiniMaxMusic3LatentAudio', 'Duration latent'], ['7', 'KSampler', 'Sampler'], ['8', 'VAEDecodeAudio', 'Decode'], ['9', 'VAEDecodeAudioTiled', 'Tiled decode'], ['10', 'ComfySwitchNode', 'Decode switch'], ['11', 'SaveAudioAdvanced', 'Audio output'],
  ].map(([id, class_type, title], index) => ({ id, class_type, title, x: 40 + (index % 4) * 280, y: 40 + Math.floor(index / 4) * 180, width: 240, height: 120, inputs: [], outputs: [] }));
  const links: WorkflowGraphLink[] = [['2', '4'], ['4', '5'], ['4', '7'], ['6', '7'], ['1', '7'], ['5', '7'], ['7', '8'], ['7', '9'], ['8', '10'], ['9', '10'], ['10', '11']].map(([from_node, to_node]) => ({ from_node, to_node }));
  nodes.find((node) => node.id === '4')!.widgets = [{ name: 'caption', value: generation.musicCaption || generation.prompt }, { name: 'generation_mode', value: generation.musicMode }, { name: 'lyrics', value: generation.musicMode === 'instrumental' ? '[instrumental]' : generation.musicMode === 'lyrics' ? generation.musicLyrics : '' }, { name: 'negative_guidance', value: generation.musicNegativePrompt }, { name: 'max_duration', value: generation.musicDurationSeconds }, { name: 'cfg_scale', value: generation.musicCfgScale }, { name: 'top_k', value: generation.musicTopK }];
  nodes.find((node) => node.id === '7')!.widgets = [{ name: 'steps', value: generation.steps }, { name: 'cfg', value: generation.musicCfgScale }, { name: 'sampler', value: generation.sampler }, { name: 'scheduler', value: generation.scheduler }];
  nodes.find((node) => node.id === '10')!.widgets = [{ name: 'tiled_decode', value: generation.musicTiledDecode }];
  return { nodes, links, width: 1400, height: 700 };
}

function directorPatchNodes(generation: GenerationState, directorFrames: number): WorkflowGraphNode[] {
  if (generation.preset.toLowerCase() !== 'ltx' || !generation.directorEnabled) return [];
  return [
    {
      id: 'director_patch',
      class_type: 'LTXDirector',
      title: 'LTX Director Timeline Patch',
      x: 330,
      y: 530,
      width: 260,
      height: 148,
      inputs: ['global_prompt', 'timeline_data', 'guide_strength', 'custom_audio'],
      widgets: [
        { name: 'duration_frames', value: directorFrames - 1 },
        { name: 'duration_seconds', value: generation.directorDuration.toFixed(2) },
        { name: 'guide_strength', value: generation.directorGuideStrength.toFixed(2) },
        { name: 'resize_method', value: generation.directorResizeMethod },
      ],
    },
    {
      id: 'director_trim',
      class_type: 'VHS_SelectImages',
      title: 'Trim Director Frames To Timeline',
      x: 1680,
      y: 265,
      width: 260,
      height: 116,
      inputs: ['image', 'indexes'],
      widgets: [{ name: 'indexes', value: `0:${directorFrames}` }],
    },
  ];
}

export function buildStudioWorkflowGraph(
  backendNodes: WorkflowGraphNode[],
  backendLinks: WorkflowGraphLink[],
  generation: GenerationState,
  alignedFrames: number,
  directorFrames: number,
) {
  const nodes = backendNodes.map((node) => patchedWorkflowNode(node, generation, alignedFrames, directorFrames));
  const links = [...backendLinks];
  const directorNodes = directorPatchNodes(generation, directorFrames);
  if (directorNodes.length) {
    nodes.push(...directorNodes);
    links.push(
      { from_node: 'director_patch', to_node: '4', type: 'PROMPT' },
      { from_node: 'director_patch', to_node: '7', type: 'DIRECTOR' },
      { from_node: '13', to_node: 'director_trim', type: 'IMAGE' },
      { from_node: 'director_trim', to_node: '14', type: 'IMAGE' },
    );
  }
  const width = Math.max(2400, ...nodes.map((node) => Number(node.x || 0) + Number(node.width || 230) + 120));
  const height = Math.max(920, ...nodes.map((node) => Number(node.y || 0) + Number(node.height || 118) + 120));
  return { nodes, links, width, height };
}

export function StudioWorkflowGraph({
  nodes,
  links,
  width,
  height,
  workflowName,
  synced,
}: {
  nodes: WorkflowGraphNode[];
  links: WorkflowGraphLink[];
  width: number;
  height: number;
  workflowName: string;
  synced: boolean;
}) {
  const nodeMap = new Map(nodes.map((node) => [node.id, node]));
  const scaledWidth = Math.max(width, 1600);
  const scaledHeight = Math.max(height, 900);

  return (
    <div className="studio-workflow-scroll">
      <div className="studio-workflow-plane" style={{ width: scaledWidth, height: scaledHeight }}>
        <div className="studio-node-group">{workflowName || 'Backend workflow'} {synced ? 'SYNCED' : 'FALLBACK'}</div>
        <svg className="studio-workflow-wires" width={scaledWidth} height={scaledHeight}>
          {links.map((link, index) => {
            const from = nodeMap.get(String(link.from_node));
            const to = nodeMap.get(String(link.to_node));
            if (!from || !to) return null;
            const x1 = Number(from.x || 0) + Number(from.width || 230);
            const y1 = Number(from.y || 0) + 34 + Number(link.from_slot || 0) * 12;
            const x2 = Number(to.x || 0);
            const y2 = Number(to.y || 0) + 34 + Number(link.to_slot || 0) * 12;
            const mid = Math.max(60, Math.abs(x2 - x1) * 0.45);
            return (
              <path
                key={`${link.from_node}-${link.to_node}-${index}`}
                className={`studio-workflow-wire wire-${workflowNodeKind(from)} wire-${workflowNodeKind(to)}`}
                d={`M ${x1} ${y1} C ${x1 + mid} ${y1}, ${x2 - mid} ${y2}, ${x2} ${y2}`}
              />
            );
          })}
        </svg>
        {nodes.map((node) => (
          <article
            className={`studio-template-node node-${workflowNodeKind(node)}`}
            key={node.id}
            style={{ left: Number(node.x || 0), top: Number(node.y || 0), width: Number(node.width || 230), minHeight: Number(node.height || 118) }}
          >
            <header><strong>{node.title || node.class_type}</strong><span>#{node.id}</span></header>
            <small>{node.class_type}</small>
            <div className="studio-node-ports">
              <span>{(node.inputs ?? []).slice(0, 4).join(' / ') || 'input'}</span>
              <span>{(node.outputs ?? []).slice(0, 3).join(' / ') || 'output'}</span>
            </div>
            <div className="studio-node-widgets">
              {(node.widgets ?? []).slice(0, 5).map((widget, index) => (
                <div key={`${node.id}-${widget.name || index}`}>
                  <span>{widget.name || `value ${index + 1}`}</span>
                  <b>{shortWorkflowValue(widget.value)}</b>
                </div>
              ))}
            </div>
          </article>
        ))}
      </div>
    </div>
  );
}

