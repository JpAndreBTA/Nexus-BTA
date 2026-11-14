import { useState, type PointerEvent } from 'react';
import { FilePlus2, Images, Redo2, Undo2 } from 'lucide-react';
import type { GenerationState, PromptRegion } from '../../stores/generationStore';

function clampRegion(region: PromptRegion): PromptRegion {
  const w = Math.max(0.03, Math.min(1, region.w));
  const h = Math.max(0.03, Math.min(1, region.h));
  return {
    ...region,
    w,
    h,
    x: Math.max(0, Math.min(1 - w, region.x)),
    y: Math.max(0, Math.min(1 - h, region.y)),
  };
}


export function IdeogramRegionEditor({
  generation,
  sourceUrl,
  previewIsVideo,
}: {
  generation: GenerationState;
  sourceUrl: string;
  previewIsVideo: boolean;
}) {
  const [activeId, setActiveId] = useState('');
  const [dragState, setDragState] = useState<{ id: string; startX: number; startY: number; x: number; y: number } | null>(null);
  const activeRegion = generation.promptRegions.find((region) => region.id === activeId) || generation.promptRegions[0];
  const aspectRatio = `${Math.max(64, generation.width)} / ${Math.max(64, generation.height)}`;

  function addRegion(type: 'obj' | 'text' = 'obj') {
    const region = {
      id: `region-${Date.now()}-${Math.round(Math.random() * 10000)}`,
      type,
      x: 0.18 + Math.min(0.18, generation.promptRegions.length * 0.04),
      y: 0.18 + Math.min(0.18, generation.promptRegions.length * 0.04),
      w: type === 'text' ? 0.42 : 0.34,
      h: type === 'text' ? 0.16 : 0.3,
      prompt: type === 'text' ? 'Localized text element' : 'Localized object or subject',
      text: type === 'text' ? 'TEXT' : '',
    };
    generation.addPromptRegion(region);
    setActiveId(region.id);
  }

  function updateRegion(id: string, updates: Partial<PromptRegion>) {
    const current = generation.promptRegions.find((region) => region.id === id);
    if (!current) return;
    generation.updatePromptRegion(id, clampRegion({ ...current, ...updates }));
  }

  function pointerPosition(event: PointerEvent, element: Element | null) {
    const rect = (element || event.currentTarget).getBoundingClientRect();
    return {
      x: (event.clientX - rect.left) / Math.max(1, rect.width),
      y: (event.clientY - rect.top) / Math.max(1, rect.height),
    };
  }

  return (
    <div className="ideogram-region-shell">
      <div className="ideogram-region-toolbar">
        <button type="button" onClick={() => addRegion('obj')}><FilePlus2 size={13} /> ADD obj</button>
        <button type="button" onClick={() => addRegion('text')}><FilePlus2 size={13} /> ADD text</button>
        <button type="button" onClick={generation.undoPromptRegion} disabled={!generation.promptRegionHistory.length}><Undo2 size={13} /></button>
        <button type="button" onClick={generation.redoPromptRegion} disabled={!generation.promptRegionFuture.length}><Redo2 size={13} /></button>
        <button type="button" onClick={generation.clearPromptRegions} disabled={!generation.promptRegions.length}>Clear</button>
      </div>
      <div
        className="ideogram-region-canvas"
        style={{ aspectRatio }}
        onPointerMove={(event) => {
          if (!dragState) return;
          const point = pointerPosition(event, event.currentTarget);
          updateRegion(dragState.id, {
            x: dragState.x + point.x - dragState.startX,
            y: dragState.y + point.y - dragState.startY,
          });
        }}
        onPointerUp={() => setDragState(null)}
        onPointerCancel={() => setDragState(null)}
      >
        {sourceUrl ? (
          previewIsVideo ? <video src={sourceUrl} muted playsInline /> : <img src={sourceUrl} alt="Ideogram regional guide" />
        ) : (
          <div className="ideogram-region-empty"><Images size={34} /><span>{generation.activity === 'img2img' ? 'Load img2img reference' : 'Regional prompt canvas'}</span></div>
        )}
        {generation.promptRegions.map((region, index) => (
          <button
            type="button"
            key={region.id}
            className={region.id === activeRegion?.id ? 'ideogram-region-box active' : 'ideogram-region-box'}
            style={{
              left: `${region.x * 100}%`,
              top: `${region.y * 100}%`,
              width: `${region.w * 100}%`,
              height: `${region.h * 100}%`,
            }}
            onClick={() => setActiveId(region.id)}
            onPointerDown={(event) => {
              const point = pointerPosition(event, event.currentTarget.parentElement);
              setActiveId(region.id);
              setDragState({ id: region.id, startX: point.x, startY: point.y, x: region.x, y: region.y });
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
          >
            <span>{String(index + 1).padStart(2, '0')}</span>
            <strong>{region.type}</strong>
            <em>{region.text || region.prompt || 'region'}</em>
            <i
              role="presentation"
              onPointerDown={(event) => event.stopPropagation()}
              onClick={(event) => {
                event.stopPropagation();
                generation.removePromptRegion(region.id);
              }}
            >
              x
            </i>
          </button>
        ))}
      </div>
      {activeRegion ? (
        <div className="ideogram-region-editor">
          <div className="two-col">
            <label className="field">
              <span>Type</span>
              <select value={activeRegion.type} onChange={(event) => updateRegion(activeRegion.id, { type: event.currentTarget.value as 'obj' | 'text' })}>
                <option value="obj">obj</option>
                <option value="text">text</option>
              </select>
            </label>
            <label className="field">
              <span>Text</span>
              <input value={activeRegion.text || ''} onChange={(event) => updateRegion(activeRegion.id, { text: event.currentTarget.value })} placeholder="literal text" />
            </label>
          </div>
          <label className="field">
            <span>Region prompt</span>
            <textarea value={activeRegion.prompt} onChange={(event) => updateRegion(activeRegion.id, { prompt: event.currentTarget.value })} placeholder="description of this region" />
          </label>
          <div className="four-col">
            <label className="field"><span>X %</span><input type="number" min={0} max={100} value={Math.round(activeRegion.x * 100)} onChange={(event) => updateRegion(activeRegion.id, { x: Number(event.currentTarget.value) / 100 })} /></label>
            <label className="field"><span>Y %</span><input type="number" min={0} max={100} value={Math.round(activeRegion.y * 100)} onChange={(event) => updateRegion(activeRegion.id, { y: Number(event.currentTarget.value) / 100 })} /></label>
            <label className="field"><span>W %</span><input type="number" min={3} max={100} value={Math.round(activeRegion.w * 100)} onChange={(event) => updateRegion(activeRegion.id, { w: Number(event.currentTarget.value) / 100 })} /></label>
            <label className="field"><span>H %</span><input type="number" min={3} max={100} value={Math.round(activeRegion.h * 100)} onChange={(event) => updateRegion(activeRegion.id, { h: Number(event.currentTarget.value) / 100 })} /></label>
          </div>
        </div>
      ) : (
        <div className="ideogram-region-editor empty">ADD creates localized prompt boxes for Ideogram 4 JSON layout guidance.</div>
      )}
    </div>
  );
}

