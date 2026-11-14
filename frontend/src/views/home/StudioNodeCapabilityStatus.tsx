import type { ComfyCapabilityReport } from '../../api/types';

/** Read-only diagnostics; offline != missing, no automatic installations. */
export function StudioNodeCapabilityStatus({
  report, count, loading, error,
}: {
  report?: ComfyCapabilityReport;
  count: number;
  loading: boolean;
  error: boolean;
}) {
  if (!count) {
    return <div className="studio-node-capabilities" role="status">No imported workflow nodes available for registration checks.</div>;
  }
  if (loading) {
    return <div className="studio-node-capabilities" role="status">Inspecting {count} node classes in the running ComfyUI…</div>;
  }
  if (error || !report || report.status === 'unavailable') {
    return <div className="studio-node-capabilities" role="status">ComfyUI node registry is unavailable. Compatibility has not been certified.</div>;
  }
  if (report.status === 'unverified' || report.status === 'partial') {
    return <div className="studio-node-capabilities missing" role="status">
      <strong>{report.required_count ?? count} workflow classes</strong>
      <span>{report.warning || 'This workflow cannot be completely verified against the current registry.'}</span>
      <small>No nodes were installed or upgraded.</small>
    </div>;
  }
  if (!report.runtime_checked || report.status === 'offline') {
    return <div className="studio-node-capabilities" role="status">ComfyUI is offline. Node compatibility is unverified; nothing was started.</div>;
  }
  return (
    <div className={report.ready ? 'studio-node-capabilities' : 'studio-node-capabilities missing'} role="status">
      <strong>{report.available.length}/{report.requested.length} registered nodes</strong>
      {report.ready
        ? <span>Current workflow classes were found in the running ComfyUI.</span>
        : <span>Missing registered classes: {report.missing.slice(0, 8).join(', ')}{report.missing.length > 8 ? ` (+${report.missing.length - 8})` : ''}. Check custom-node startup logs before installing anything.</span>
      }
      <small>Classes are read from the imported workflow JSON. Registration alone does not prove inputs, models, VRAM or execution.</small>
    </div>
  );
}
