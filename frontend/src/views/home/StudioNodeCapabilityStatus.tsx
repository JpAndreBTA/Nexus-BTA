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
      <small>Registration alone does not prove model availability or workflow execution.</small>
    </div>
  );
}
