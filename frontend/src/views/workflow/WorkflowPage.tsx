import { useEffect, useMemo, useState } from 'react';
import { FileInput, GitBranch, LoaderCircle, LocateFixed, Upload, Workflow, ZoomIn, ZoomOut } from 'lucide-react';
import { useMutation } from '@tanstack/react-query';

import { nexusApi } from '../../api/nexusClient';
import { useWorkflowAnalysisQuery, useWorkflowsQuery } from '../../api/queries';
import type { WorkflowAnalysis, WorkflowGraphNode } from '../../api/types';
import { queryClient } from '../../shared/queryClient';
import { useGenerationStore } from '../../stores/generationStore';

function nodeLabel(node: WorkflowGraphNode) {
  return node.title || node.class_type || node.id;
}

function widgetValue(node: WorkflowGraphNode, widgetName: string) {
  return node.widgets?.find((widget) => widget.name === widgetName)?.value;
}

function numberValue(value: unknown, fallback: number) {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function nodeKind(node: WorkflowGraphNode) {
  const text = `${node.class_type} ${node.title || ''}`.toLowerCase();
  if (text.includes('control')) return 'control';
  if (text.includes('checkpoint') || text.includes('unet') || text.includes('model') || text.includes('loader')) return 'model';
  if (text.includes('prompt') || text.includes('text') || text.includes('clip')) return 'text';
  if (text.includes('sampler') || text.includes('scheduler') || text.includes('latent')) return 'sample';
  if (text.includes('vae') || text.includes('decode') || text.includes('image') || text.includes('video')) return 'media';
  return 'utility';
}

export function WorkflowPage() {
  const workflows = useWorkflowsQuery();
  const generation = useGenerationStore();
  const [selectedId, setSelectedId] = useState('');
  const [editableNodes, setEditableNodes] = useState<WorkflowGraphNode[]>([]);
  const [localError, setLocalError] = useState('');
  const [selectedNodeId, setSelectedNodeId] = useState('');
  const [zoom, setZoom] = useState(0.82);

  useEffect(() => {
    if (!selectedId && workflows.data?.[0]) {
      const preferred = workflows.data.find((workflow) => workflow.id === generation.workflowId);
      setSelectedId((preferred || workflows.data[0]).id);
    }
  }, [generation.workflowId, selectedId, workflows.data]);

  const analysis = useWorkflowAnalysisQuery(selectedId);
  const graph = analysis.data?.visual_graph;
  const graphNodes = useMemo(() => graph?.nodes ?? [], [graph?.nodes]);
  const nodes = editableNodes.length ? editableNodes : graphNodes;
  const links = graph?.links ?? [];
  const nodeMap = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const missing = new Set(analysis.data?.missing_nodes ?? []);
  const selectedWorkflow = workflows.data?.find((workflow) => workflow.id === selectedId) || analysis.data?.workflow;
  const graphWidth = Math.max(900, Number(graph?.width || 1200));
  const graphHeight = Math.max(520, Number(graph?.height || 680));
  const selectedNode = nodes.find((node) => node.id === selectedNodeId) || nodes[0];

  useEffect(() => {
    setEditableNodes(graphNodes.map((node) => ({ ...node, widgets: node.widgets?.map((widget) => ({ ...widget })) ?? [] })));
    setSelectedNodeId(graphNodes[0]?.id || '');
    setLocalError('');
  }, [graphNodes]);

  const loadMutation = useMutation({
    mutationFn: (file: File) => nexusApi.loadWorkflow(file),
    onSuccess: (loaded) => {
      queryClient.setQueryData(['workflow-analysis', loaded.workflow.id], loaded);
      setSelectedId(loaded.workflow.id);
      setLocalError('');
    },
    onError: (error) => setLocalError(error instanceof Error ? error.message : 'Workflow load failed.'),
  });

  const importMutation = useMutation({
    mutationFn: (file: File) => nexusApi.importWorkflow(file),
    onSuccess: (loaded) => {
      queryClient.setQueryData(['workflow-analysis', loaded.workflow.id], loaded);
      queryClient.invalidateQueries({ queryKey: ['workflows'] });
      setSelectedId(loaded.workflow.id);
      setLocalError('');
    },
    onError: (error) => setLocalError(error instanceof Error ? error.message : 'Workflow import failed.'),
  });

  function syncWidgetToStudio(node: WorkflowGraphNode, name: string, value: unknown) {
    const lowerName = name.toLowerCase();
    const role = `${node.title || ''} ${node.class_type || ''}`.toLowerCase();
    const stringValue = String(value ?? '');

    if (lowerName === 'text') {
      if (role.includes('negative')) {
        generation.setNegativePrompt(stringValue);
      } else if (role.includes('positive') || role.includes('prompt') || node.class_type.toLowerCase().includes('text')) {
        generation.setPrompt(stringValue);
      }
      return;
    }
    if (lowerName === 'width') {
      generation.setSize(numberValue(value, generation.width), generation.height);
      return;
    }
    if (lowerName === 'height') {
      generation.setSize(generation.width, numberValue(value, generation.height));
      return;
    }
    if (lowerName === 'seed') {
      generation.setSeed(numberValue(value, generation.seed));
      return;
    }
    if (lowerName === 'steps') {
      generation.setSteps(numberValue(value, generation.steps));
      return;
    }
    if (lowerName === 'cfg') {
      generation.setCfg(numberValue(value, generation.cfg));
      return;
    }
    if (lowerName === 'sampler_name') {
      generation.setSampler(stringValue);
      return;
    }
    if (lowerName === 'scheduler') {
      generation.setScheduler(stringValue);
      return;
    }
    if (lowerName === 'denoise') {
      generation.setDenoise(numberValue(value, generation.denoise));
      return;
    }
    if (lowerName === 'ckpt_name' || lowerName === 'unet_name') {
      generation.setModel(stringValue, stringValue.split(/[\\/]/).pop() || stringValue);
    }
  }

  function updateWidget(node: WorkflowGraphNode, name: string, value: string) {
    setEditableNodes((current) =>
      current.map((item) =>
        item.id === node.id
          ? {
              ...item,
              widgets: (item.widgets ?? []).map((widget) => (widget.name === name ? { ...widget, value } : widget)),
            }
          : item,
      ),
    );
    syncWidgetToStudio(node, name, value);
  }

  function activateWorkflow(workflow: WorkflowAnalysis['workflow']) {
    generation.setWorkflow(workflow.id, workflow.name);
    editableNodes.forEach((node) => {
      (node.widgets ?? []).forEach((widget) => {
        if (widget.name) syncWidgetToStudio(node, widget.name, widget.value);
      });
    });
  }

  return (
    <section className="page workflow-layout">
      <header className="page-header">
        <div>
          <p className="eyebrow">ComfyUI</p>
          <h1>Workflow Graph</h1>
        </div>
        <span className="status-pill ok">
          <GitBranch size={14} />
          {workflows.data?.length ?? 0} workflows
        </span>
      </header>

      <div className="workflow-columns">
        <aside className="surface workflow-sidebar">
          <label className="field">
            <span>Workflow</span>
            <select value={selectedId} onChange={(event) => setSelectedId(event.currentTarget.value)}>
              {(workflows.data ?? []).map((workflow) => (
                <option key={workflow.id} value={workflow.id}>
                  {workflow.name}
                </option>
              ))}
            </select>
          </label>

          {selectedWorkflow && (
            <div className="workflow-summary">
              <strong>{selectedWorkflow.name}</strong>
              <span>{selectedWorkflow.format.toUpperCase()} / {selectedWorkflow.node_count} nodes</span>
              <span>{selectedWorkflow.path}</span>
              <button
                className="primary-button"
                type="button"
                onClick={() => activateWorkflow(selectedWorkflow)}
              >
                <Workflow size={14} />
                Activate & Sync
              </button>
            </div>
          )}

          <div className="workflow-file-actions">
            <label className="flat-button">
              <FileInput size={14} />
              Load JSON
              <input
                type="file"
                accept=".json,application/json"
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  if (file) loadMutation.mutate(file);
                  event.currentTarget.value = '';
                }}
              />
            </label>
            <label className="flat-button">
              <Upload size={14} />
              Import
              <input
                type="file"
                accept=".json,application/json"
                onChange={(event) => {
                  const file = event.currentTarget.files?.[0];
                  if (file) importMutation.mutate(file);
                  event.currentTarget.value = '';
                }}
              />
            </label>
          </div>

          {(localError || loadMutation.isPending || importMutation.isPending) && (
            <p className="compact-note">{localError || 'Reading workflow JSON...'}</p>
          )}
          <p className="compact-note">Widget edits sync Studio controls. Executable workflow override stays disabled until the backend exposes full untruncated workflow JSON.</p>

          <div className="workflow-health">
            <span>Available nodes</span>
            <strong>{analysis.data?.available_nodes ?? 0}</strong>
            <span>Missing nodes</span>
            <strong>{analysis.data?.missing_nodes.length ?? 0}</strong>
            <span>Dependency targets</span>
            <strong>{analysis.data?.dependency_targets.length ?? 0}</strong>
          </div>

          {analysis.data?.missing_nodes.length ? (
            <div className="missing-node-list">
              {analysis.data.missing_nodes.slice(0, 12).map((node) => (
                <span key={node}>{node}</span>
              ))}
            </div>
          ) : (
            <p className="compact-note">No missing workflow nodes reported.</p>
          )}

          {selectedNode && (
            <div className="workflow-inspector">
              <strong>{nodeLabel(selectedNode)}</strong>
              <span>{selectedNode.class_type}</span>
              <span>{(selectedNode.inputs ?? []).length} inputs / {(selectedNode.outputs ?? []).length} outputs</span>
              {(selectedNode.widgets ?? []).slice(0, 8).map((widget) => (
                <label className="field" key={`inspect-${selectedNode.id}-${widget.name}`}>
                  <span>{widget.name}</span>
                  <input value={String(widget.value ?? '')} onChange={(event) => updateWidget(selectedNode, String(widget.name || ''), event.currentTarget.value)} />
                </label>
              ))}
            </div>
          )}
        </aside>

        <main className="surface workflow-canvas">
          <div className="workflow-toolbar">
            <button className="mini-button" type="button" onClick={() => setZoom((value) => Math.max(0.45, Number((value - 0.1).toFixed(2))))} title="Zoom out">
              <ZoomOut size={14} />
            </button>
            <span>{Math.round(zoom * 100)}%</span>
            <button className="mini-button" type="button" onClick={() => setZoom((value) => Math.min(1.4, Number((value + 0.1).toFixed(2))))} title="Zoom in">
              <ZoomIn size={14} />
            </button>
            <button className="mini-button" type="button" onClick={() => setZoom(0.82)} title="Reset zoom">
              <LocateFixed size={14} />
            </button>
          </div>
          {analysis.isFetching && (
            <div className="workflow-loading">
              <LoaderCircle className="spin" size={28} />
            </div>
          )}
          <div className="workflow-plane" style={{ width: graphWidth, height: graphHeight, transform: `scale(${zoom})` }}>
            <svg className="workflow-wires" width={graphWidth} height={graphHeight}>
              {links.map((link, index) => {
                const from = nodeMap.get(link.from_node);
                const to = nodeMap.get(link.to_node);
                if (!from || !to) return null;
                const x1 = Number(from.x || 0) + Number(from.width || 220);
                const y1 = Number(from.y || 0) + Number(from.height || 118) / 2;
                const x2 = Number(to.x || 0);
                const y2 = Number(to.y || 0) + Number(to.height || 118) / 2;
                return <path key={`${link.from_node}-${link.to_node}-${index}`} d={`M ${x1} ${y1} C ${x1 + 70} ${y1}, ${x2 - 70} ${y2}, ${x2} ${y2}`} />;
              })}
            </svg>
            {nodes.map((node) => {
              const isMissing = missing.has(node.class_type);
              const selected = selectedNodeId === node.id;
              return (
                <article
                  className={['workflow-node-card', `node-${nodeKind(node)}`, isMissing ? 'missing' : '', selected ? 'selected' : ''].filter(Boolean).join(' ')}
                  key={node.id}
                  onClick={() => setSelectedNodeId(node.id)}
                  style={{
                    left: Number(node.x || 0),
                    top: Number(node.y || 0),
                    width: Math.max(190, Math.min(360, Number(node.width || 220))),
                  }}
                >
                  <header>
                    <strong>{nodeLabel(node)}</strong>
                    <span>#{node.id}</span>
                  </header>
                  <p>{node.class_type}</p>
                  {(node.widgets ?? []).slice(0, 4).map((widget) => (
                    <div className="workflow-widget" key={`${node.id}-${widget.name}`}>
                      <span>{widget.name}</span>
                      <input value={String(widget.value ?? '')} onChange={(event) => updateWidget(node, String(widget.name || ''), event.currentTarget.value)} />
                    </div>
                  ))}
                  {widgetValue(node, 'text') !== undefined && <em className="workflow-sync-badge">syncs prompt</em>}
                </article>
              );
            })}
          </div>
        </main>
      </div>
    </section>
  );
}
