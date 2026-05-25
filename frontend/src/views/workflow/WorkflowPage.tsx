import { useEffect, useMemo, useState } from 'react';
import { GitBranch, LoaderCircle, Workflow } from 'lucide-react';

import { useWorkflowAnalysisQuery, useWorkflowsQuery } from '../../api/queries';
import type { WorkflowGraphNode } from '../../api/types';
import { useGenerationStore } from '../../stores/generationStore';

function nodeLabel(node: WorkflowGraphNode) {
  return node.title || node.class_type || node.id;
}

export function WorkflowPage() {
  const workflows = useWorkflowsQuery();
  const generation = useGenerationStore();
  const [selectedId, setSelectedId] = useState('');

  useEffect(() => {
    if (!selectedId && workflows.data?.[0]) {
      const preferred = workflows.data.find((workflow) => workflow.id === generation.workflowId);
      setSelectedId((preferred || workflows.data[0]).id);
    }
  }, [generation.workflowId, selectedId, workflows.data]);

  const analysis = useWorkflowAnalysisQuery(selectedId);
  const graph = analysis.data?.visual_graph;
  const nodes = useMemo(() => graph?.nodes ?? [], [graph?.nodes]);
  const links = graph?.links ?? [];
  const nodeMap = useMemo(() => new Map(nodes.map((node) => [node.id, node])), [nodes]);
  const missing = new Set(analysis.data?.missing_nodes ?? []);
  const selectedWorkflow = workflows.data?.find((workflow) => workflow.id === selectedId) || analysis.data?.workflow;
  const graphWidth = Math.max(900, Number(graph?.width || 1200));
  const graphHeight = Math.max(520, Number(graph?.height || 680));

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
                onClick={() => generation.setWorkflow(selectedWorkflow.id, selectedWorkflow.name)}
              >
                <Workflow size={14} />
                Activate
              </button>
            </div>
          )}

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
        </aside>

        <main className="surface workflow-canvas">
          {analysis.isFetching && (
            <div className="workflow-loading">
              <LoaderCircle className="spin" size={28} />
            </div>
          )}
          <div className="workflow-plane" style={{ width: graphWidth, height: graphHeight }}>
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
              return (
                <article
                  className={isMissing ? 'workflow-node-card missing' : 'workflow-node-card'}
                  key={node.id}
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
                  {(node.widgets ?? []).slice(0, 3).map((widget) => (
                    <div className="workflow-widget" key={`${node.id}-${widget.name}`}>
                      <span>{widget.name}</span>
                      <em>{String(widget.value ?? '')}</em>
                    </div>
                  ))}
                </article>
              );
            })}
          </div>
        </main>
      </div>
    </section>
  );
}
