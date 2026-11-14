import type { WorkflowInputReport, ModelSourceAuditReport } from '../../api/types';

function ModelSourceAuditStatus({ report }: { report: ModelSourceAuditReport }) {
  return (
    <details>
      <summary>Local model files: {report.found + report.multiple_matches} found · {report.not_found} not found · {report.unverified} unverified</summary>
      <small>{report.note} Checked {report.roots_checked} configured roots and {report.entries_scanned} directory entries{report.truncated ? ' (scan truncated)' : ''}.</small>
      <ul>
        {report.entries.map((entry, index) => (
          <li key={`${entry.node_id}-${entry.input}-${index}`}>
            {entry.class_type} #{entry.node_id} / {entry.input}: {entry.status}
            {entry.sources.length > 0 ? ` · ${entry.sources.join(', ')}` : ''}
          </li>
        ))}
      </ul>
    </details>
  );
}


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
  if (report.status === 'offline') return (
    <div className="studio-node-capabilities" role="status">
      <span>Input ABI unverified: ComfyUI is offline.</span>
      {report.model_sources && <ModelSourceAuditStatus report={report.model_sources} />}
    </div>
  );
  if (report.status === 'ui_format' || report.status === 'unsupported') return (
    <div className="studio-node-capabilities" role="status">Input ABI not inspected: {report.note}</div>
  );
  const modelSourceReview = Boolean(report.model_sources && (
    report.model_sources.not_found > 0
    || report.model_sources.unverified > 0
    || report.model_sources.multiple_matches > 0
    || report.model_sources.truncated
  ));
  const requiresReview = report.status !== 'static_valid' || modelSourceReview;
  return (
    <div className={requiresReview ? 'studio-node-capabilities missing' : 'studio-node-capabilities'} role="status">
      <strong>{modelSourceReview ? 'Workflow preflight needs review' : requiresReview ? 'Input ABI needs review' : 'Static input checks passed'}</strong>
      <span>{report.node_count} nodes · {report.missing_classes} missing classes · {report.broken_links} link issues</span>
      <span>{report.missing_required_inputs} potentially missing inputs · {report.choice_mismatches} choice mismatches · {report.unverified_inputs} unverified inputs</span>
      <span>{report.model_choices_unavailable ?? 0} / {report.model_choices_checked ?? 0} model selections not advertised by the current registry</span>
      {report.schema_signature && <small>Comfy node schema signature: {report.schema_signature} (changes when registered input menus or ABI change)</small>}
      {report.model_sources && <ModelSourceAuditStatus report={report.model_sources} />}
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
