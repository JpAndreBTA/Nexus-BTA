import type { WorkflowInputReport } from '../../api/types';

/** Static advisory only: generator patches and actual executions may differ. */
export function StudioWorkflowInputStatus({
  report, loading, error,
}: {
  report?: WorkflowInputReport;
  loading: boolean;
  error: boolean;
}) {
  if (loading) return <div className="studio-node-capabilities" role="status">Inspecting saved workflow inputs…</div>;
  if (error || !report || report.status === 'unavailable') {
    return <div className="studio-node-capabilities" role="status">Workflow input compatibility is unverified; node schemas are unavailable.</div>;
  }
  if (report.status === 'offline') return <div className="studio-node-capabilities" role="status">Input ABI unverified: ComfyUI is offline.</div>;
  if (report.status === 'ui_format' || report.status === 'unsupported') return (
    <div className="studio-node-capabilities" role="status">Input ABI not inspected: {report.note}</div>
  );
  const requiresReview = report.status !== 'static_valid';
  return (
    <div className={requiresReview ? 'studio-node-capabilities missing' : 'studio-node-capabilities'} role="status">
      <strong>{requiresReview ? 'Input ABI needs review' : 'Static input checks passed'}</strong>
      <span>{report.node_count} nodes · {report.missing_classes} missing classes · {report.broken_links} link issues</span>
      <span>{report.missing_required_inputs} potentially missing inputs · {report.choice_mismatches} choice mismatches · {report.unverified_inputs} unverified inputs</span>
      {report.findings.length > 0 && (
        <details>
          <summary>Inspect first {report.findings.length} findings{report.truncated_findings ? ' (truncated)' : ''}</summary>
          <ul>
            {report.findings.slice(0, 16).map((finding, index) => (
              <li key={`${finding.node_id}-${finding.input}-${index}`}>{finding.kind}: {finding.class_type} #{finding.node_id}{finding.input ? ` / ${finding.input}` : ''}</li>
            ))}
          </ul>
        </details>
      )}
      <small>{report.note} No generation or dependency installation was attempted.</small>
    </div>
  );
}
