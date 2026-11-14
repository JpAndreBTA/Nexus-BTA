import { useEffect, useRef, useState } from 'react';
import { Link } from '@tanstack/react-router';
import { Download, Search, X } from 'lucide-react';
import { nexusApi } from '../../api/nexusClient';
import type { CivitaiModelItem } from '../../api/types';

// Optional modal chunk: loaded only when the user opens Civitai.
// Download jobs live on the backend; closing the modal does not cancel a started download.
function civitaiBaseModel(preset: string) {
  const values: Record<string, string> = {
    SD: 'SD 1.5',
    SDXL: 'SDXL 1.0',
    Flux: 'Flux.1 D',
    Qwen: 'Qwen Image',
    ZImageTurbo: 'ZImageTurbo',
    Wan: 'WAN 2.2',
    LTX: 'LTXV 2.3',
  };
  return values[preset] || preset;
}

function isVideoPreview(url: string | undefined) {
  return /\.(mp4|webm|mov)(\?|$)/i.test(String(url || ''));
}


export function CivitaiModal({ preset, onClose }: { preset: string; onClose: () => void }) {
  const civitaiDialogRef = useRef<HTMLDivElement | null>(null);
  const [civitaiFilter, setCivitaiFilter] = useState('All Types');
  const [civitaiDownloadJobId, setCivitaiDownloadJobId] = useState('');
  const [civitaiDownloadStatus, setCivitaiDownloadStatus] = useState('');
  const [civitaiDownloading, setCivitaiDownloading] = useState(false);
  const [civitaiQuery, setCivitaiQuery] = useState('');
  const [civitaiResults, setCivitaiResults] = useState<CivitaiModelItem[]>([]);
  const [civitaiStatus, setCivitaiStatus] = useState('');

  useEffect(() => {
    if (!civitaiDownloadJobId) return;
    let disposed = false;
    const poll = async () => {
      try {
        const job = await nexusApi.civitaiDownloadJob(civitaiDownloadJobId);
        if (disposed) return;
        setCivitaiDownloadStatus(`${job.message || job.status} (${Math.round(job.progress || 0)}%)`);
        if (['completed', 'failed', 'cancelled'].includes(job.status)) {
          setCivitaiDownloading(false);
          setCivitaiDownloadJobId('');
        }
      } catch (error) {
        if (disposed) return;
        setCivitaiDownloading(false);
        setCivitaiDownloadJobId('');
        setCivitaiDownloadStatus(error instanceof Error ? error.message : 'Download status unavailable.');
      }
    };
    void poll();
    const timer = window.setInterval(() => { void poll(); }, 1500);
    return () => { disposed = true; window.clearInterval(timer); };
  }, [civitaiDownloadJobId]);

  useEffect(() => {
    const previousFocus = document.activeElement instanceof HTMLElement ? document.activeElement : null;
    const dialog = civitaiDialogRef.current;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    dialog?.querySelector<HTMLInputElement>('input')?.focus();

    function handleDialogKeydown(event: KeyboardEvent) {
      if (event.key === 'Escape') {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== 'Tab' || !dialog) return;
      const tabbable = Array.from(dialog.querySelectorAll<HTMLElement>(
        'button:not([disabled]), a[href], input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])'
      )).filter((element) => element.getClientRects().length > 0);
      if (!tabbable.length) {
        event.preventDefault();
        return;
      }
      const first = tabbable[0];
      const last = tabbable[tabbable.length - 1];
      if (event.shiftKey && (document.activeElement === first || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
        event.preventDefault();
        first.focus();
      }
    }

    document.addEventListener('keydown', handleDialogKeydown);
    return () => {
      document.removeEventListener('keydown', handleDialogKeydown);
      document.body.style.overflow = previousOverflow;
      previousFocus?.focus();
    };
  }, [onClose]);

  async function downloadCivitaiModel(url: string) {
    if (!url || civitaiDownloading) return;
    setCivitaiDownloading(true);
    setCivitaiDownloadStatus('Starting download…');
    try {
      const job = await nexusApi.startCivitaiDownload(url, preset);
      setCivitaiDownloadJobId(job.job_id);
    } catch (error) {
      setCivitaiDownloading(false);
      setCivitaiDownloadStatus(error instanceof Error ? error.message : 'Download could not start.');
    }
  }

  async function runCivitaiLookup() {
    const query = civitaiQuery.trim();
    setCivitaiStatus('Searching Civitai...');
    try {
      if (/^https?:\/\//i.test(query)) {
        const resolved = await nexusApi.civitaiResolve({ url: query, preset: preset, target_kind: 'auto' });
        setCivitaiResults([
          {
            id: String(resolved.model_id || resolved.version_id || query),
            name: String(resolved.model_name || 'Resolved Civitai asset'),
            type: String(resolved.model_type || resolved.target_kind || 'Model'),
            creator: String(resolved.creator || ''),
            preview: String(resolved.preview || ''),
            versions: [{ name: String(resolved.version_name || ''), base_model: String(resolved.base_model || ''), url: String(resolved.url || query), file_name: String(resolved.file_name || '') }],
          },
        ]);
        setCivitaiStatus('URL resolved. Download wiring is available in the backend.');
        return;
      }
      const result = await nexusApi.civitaiSearch({
        query,
        types: ({ 'All Types': '', Checkpoints: 'Checkpoint', LoRAs: 'LORA', Embeddings: 'TextualInversion' } as Record<string, string>)[civitaiFilter] || '',
        base_model: civitaiBaseModel(preset),
        sort: 'Most Downloaded',
        period: 'AllTime',
        nsfw: true,
        limit: 20,
      });
      setCivitaiResults(result.items || []);
      setCivitaiStatus(`${result.items?.length ?? 0} result(s)`);
    } catch (error) {
      setCivitaiStatus(error instanceof Error ? error.message : 'Civitai lookup failed.');
    }
  }


  return (
        <div className="modal-backdrop" role="dialog" aria-modal="true" aria-label="Nexus Civitai downloader">
          <div className="modal-panel civitai-modal" ref={civitaiDialogRef}>
            <header className="modal-header">
              <strong>Nexus Civitai Downloader</strong>
              <button className="icon-button" type="button" onClick={() => onClose()} title="Close Civitai downloader">
                <X size={15} />
              </button>
            </header>
            <div className="modal-search-row">
              <Search size={15} />
              <input value={civitaiQuery} onChange={(event) => setCivitaiQuery(event.currentTarget.value)} placeholder="Search terms or paste a Civitai URL..." />
              <button className="primary-button" type="button" onClick={runCivitaiLookup}>Resolve URL</button>
            </div>
            <div className="modal-filter-row">
              {['All Types', 'Checkpoints', 'LoRAs', 'Embeddings'].map((label) => (
                <button className={civitaiFilter === label ? 'mini-button active' : 'mini-button'} type="button" key={label} aria-pressed={civitaiFilter === label} onClick={() => setCivitaiFilter(label)}>{label}</button>
              ))}
              <span>Base: {civitaiBaseModel(preset)}</span>
              <span>Order: Most Downloaded</span>
            </div>
            <div className="civitai-card-grid">
              {!civitaiResults.length && <p className="muted" role="status">No models loaded. Enter a search or a Civitai URL to show verified results.</p>}
              {civitaiResults.map((item) => (
                <article className="civitai-card" key={`${item.id || item.name}`}>
                  {item.preview && !isVideoPreview(item.preview) ? <img src={item.preview} alt={item.name} loading="lazy" decoding="async" /> : <div><span>{item.versions?.[0]?.base_model || item.type}</span></div>}
                  <strong>{item.name}</strong>
                  <small>{item.creator ? `by ${item.creator}` : item.type} {item.versions?.[0]?.file_name ? `/ ${item.versions[0].file_name}` : ''}</small>
                  <button type="button" className="mini-button" disabled={!item.versions?.[0]?.url || civitaiDownloading}
                    onClick={() => { void downloadCivitaiModel(item.versions?.[0]?.url || ''); }}>
                    <Download size={13} /> Download
                  </button>
                </article>
              ))}
            </div>
            <footer className="modal-footer">
              <span role="status" aria-live="polite">{civitaiDownloadStatus || civitaiStatus || 'Search uses the local backend Civitai bridge.'}</span>
              <Link className="flat-button" to="/models" onClick={() => onClose()}>Open Local Models</Link>
            </footer>
          </div>
        </div>

  );
}
