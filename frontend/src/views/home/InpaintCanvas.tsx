import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { Brush, Eraser, Pencil, RotateCcw, RotateCw, Trash2 } from 'lucide-react';

interface InpaintCanvasProps {
  image: string;
  brushSize: number;
  onBrushSizeChange: (brushSize: number) => void;
  onMaskChange: (mask: string | null) => void;
}

import { MAX_EDIT_EDGE, binaryMaskPixels, fittedMaskSize, historyLimit } from './maskPixels';
type PaintTool = 'mask' | 'eraser' | 'sketch';
type Point = { x: number; y: number };
type History = { undo: ImageData[]; redo: ImageData[] };

function pointFromEvent(event: PointerEvent<HTMLCanvasElement>, canvas: HTMLCanvasElement): Point {
  const rect = canvas.getBoundingClientRect();
  const x = ((event.clientX - rect.left) / Math.max(1, rect.width)) * canvas.width;
  const y = ((event.clientY - rect.top) / Math.max(1, rect.height)) * canvas.height;
  return {
    x: Math.min(canvas.width, Math.max(0, x)),
    y: Math.min(canvas.height, Math.max(0, y)),
  };
}

// Preserves the legacy black/white PNG contract, but reuses a single export canvas.
function exportMask(sourceCanvas: HTMLCanvasElement, exportCanvas: HTMLCanvasElement): string | null {
  const source = sourceCanvas.getContext('2d', { willReadFrequently: true });
  if (!source) return null;

  const pixels = source.getImageData(0, 0, sourceCanvas.width, sourceCanvas.height);
  if (!binaryMaskPixels(pixels.data)) return null;
  if (exportCanvas.width !== sourceCanvas.width || exportCanvas.height !== sourceCanvas.height) {
    exportCanvas.width = sourceCanvas.width;
    exportCanvas.height = sourceCanvas.height;
  }
  const target = exportCanvas.getContext('2d');
  if (!target) return null;
  target.putImageData(pixels, 0, 0);
  return exportCanvas.toDataURL('image/png');
}

export function InpaintCanvas({ image, brushSize, onBrushSizeChange, onMaskChange }: InpaintCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const exportCanvasRef = useRef<HTMLCanvasElement | null>(null);
  const activePointerIdRef = useRef<number | null>(null);
  const lastPointRef = useRef<Point | null>(null);
  const historyRef = useRef<History>({ undo: [], redo: [] });
  const [historyCounts, setHistoryCounts] = useState({ undo: 0, redo: 0 });
  const [tool, setTool] = useState<PaintTool>('mask');
  const [sketchColor, setSketchColor] = useState('#ff3b3b');
  const [canvasSize, setCanvasSize] = useState({ width: MAX_EDIT_EDGE, height: MAX_EDIT_EDGE });

  useEffect(() => {
    let disposed = false;
    const source = new Image();
    source.onload = () => {
      if (disposed) return;
      setCanvasSize(fittedMaskSize(source.naturalWidth, source.naturalHeight));
    };
    source.src = image;
    return () => {
      disposed = true;
      source.onload = null;
      source.src = '';
    };
  }, [image]);

  // Reset if the source changes, even when the next image has the same dimensions.
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    activePointerIdRef.current = null;
    lastPointRef.current = null;
    historyRef.current = { undo: [], redo: [] };
    setHistoryCounts({ undo: 0, redo: 0 });
    onMaskChange(null);
  }, [image, canvasSize.width, canvasSize.height, onMaskChange]);

  useEffect(() => () => {
    historyRef.current = { undo: [], redo: [] };
    if (exportCanvasRef.current) {
      exportCanvasRef.current.width = 0;
      exportCanvasRef.current.height = 0;
      exportCanvasRef.current = null;
    }
  }, []);

  function snapshot(): ImageData | null {
    const canvas = canvasRef.current;
    return canvas?.getContext('2d', { willReadFrequently: true })?.getImageData(0, 0, canvas.width, canvas.height) ?? null;
  }

  function syncHistory() {
    const canvas = canvasRef.current;
    const history = historyRef.current;
    // Count both stacks against a shared memory budget, not just number of actions.
    const limit = historyLimit(canvas?.width || 1, canvas?.height || 1);
    while (history.undo.length + history.redo.length > limit) {
      if (history.undo.length > 1 || !history.redo.length) history.undo.shift();
      else history.redo.shift();
    }
    setHistoryCounts({ undo: history.undo.length, redo: history.redo.length });
  }

  function restore(snapshotValue: ImageData) {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context || snapshotValue.width !== canvas.width || snapshotValue.height !== canvas.height) return;
    context.putImageData(snapshotValue, 0, 0);
    publishMask();
  }

  function publishMask() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    if (!exportCanvasRef.current) exportCanvasRef.current = document.createElement('canvas');
    onMaskChange(exportMask(canvas, exportCanvasRef.current));
  }

  function pushUndo() {
    const captured = snapshot();
    if (!captured) return;
    historyRef.current.undo.push(captured);
    historyRef.current.redo = [];
    syncHistory();
  }

  function drawTo(point: Point) {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const lastPoint = lastPointRef.current;
    if (!canvas || !context || !lastPoint) return;
    context.save();
    context.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
    context.strokeStyle = tool === 'sketch' ? sketchColor : 'rgba(255,255,255,0.9)';
    context.lineWidth = brushSize;
    context.lineCap = 'round';
    context.lineJoin = 'round';
    context.beginPath();
    context.moveTo(lastPoint.x, lastPoint.y);
    context.lineTo(point.x, point.y);
    context.stroke();
    context.restore();
    lastPointRef.current = point;
  }

  function drawDot(point: Point) {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    context.save();
    context.globalCompositeOperation = tool === 'eraser' ? 'destination-out' : 'source-over';
    context.fillStyle = tool === 'sketch' ? sketchColor : 'rgba(255,255,255,0.9)';
    context.beginPath();
    context.arc(point.x, point.y, Math.max(1, brushSize / 2), 0, Math.PI * 2);
    context.fill();
    context.restore();
  }

  function finishDrawing(pointerId: number) {
    if (activePointerIdRef.current !== pointerId) return;
    activePointerIdRef.current = null;
    lastPointRef.current = null;
    publishMask();
  }

  function clearMask() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    pushUndo();
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    onMaskChange(null);
  }

  function undo() {
    if (activePointerIdRef.current !== null) return;
    const previous = historyRef.current.undo.pop();
    if (!previous) return;
    const current = snapshot();
    if (current) historyRef.current.redo.push(current);
    restore(previous);
    syncHistory();
  }

  function redo() {
    if (activePointerIdRef.current !== null) return;
    const next = historyRef.current.redo.pop();
    if (!next) return;
    const current = snapshot();
    if (current) historyRef.current.undo.push(current);
    restore(next);
    syncHistory();
  }

  return (
    <section className="inpaint-editor">
      <div className="inpaint-toolbar">
        <div className="tool-segment">
          <button className={tool === 'mask' ? 'active' : ''} aria-label="Mask brush" type="button" onClick={() => setTool('mask')} title="Mask brush">
            <Brush size={14} />
          </button>
          <button className={tool === 'eraser' ? 'active' : ''} aria-label="Erase mask" type="button" onClick={() => setTool('eraser')} title="Erase mask">
            <Eraser size={14} />
          </button>
          <button className={tool === 'sketch' ? 'active' : ''} aria-label="Inpaint sketch" type="button" onClick={() => setTool('sketch')} title="Inpaint sketch">
            <Pencil size={14} />
          </button>
        </div>
        <label className="field">
          <span>Brush {brushSize}px</span>
          <input type="range" min={8} max={160} step={2} value={brushSize} onChange={(event) => onBrushSizeChange(Number(event.currentTarget.value))} />
        </label>
        <label className="color-chip" title="Sketch color">
          <input type="color" value={sketchColor} onChange={(event) => setSketchColor(event.currentTarget.value)} />
          <span style={{ background: sketchColor }} />
        </label>
        <button className="mini-button" type="button" onClick={undo} disabled={!historyCounts.undo} aria-label="Undo mask change" title="Undo">
          <RotateCcw size={14} />
        </button>
        <button className="mini-button" type="button" onClick={redo} disabled={!historyCounts.redo} aria-label="Redo mask change" title="Redo">
          <RotateCw size={14} />
        </button>
        <button className="mini-button" type="button" onClick={clearMask} aria-label="Clear inpaint mask" title="Clear inpaint mask">
          <Trash2 size={14} />
        </button>
      </div>
      <div className="inpaint-stage" style={{ aspectRatio: `${canvasSize.width} / ${canvasSize.height}` }}>
        <img src={image} alt="Inpaint reference" draggable={false} />
        <canvas
          ref={canvasRef}
          aria-label="Inpaint mask canvas"
          onPointerDown={(event) => {
            if (activePointerIdRef.current !== null || (event.pointerType === 'mouse' && event.button !== 0)) return;
            event.currentTarget.setPointerCapture(event.pointerId);
            pushUndo();
            activePointerIdRef.current = event.pointerId;
            const point = pointFromEvent(event, event.currentTarget);
            lastPointRef.current = point;
            drawDot(point);
          }}
          onPointerMove={(event) => {
            if (activePointerIdRef.current !== event.pointerId) return;
            drawTo(pointFromEvent(event, event.currentTarget));
          }}
          onPointerUp={(event) => finishDrawing(event.pointerId)}
          onPointerCancel={(event) => finishDrawing(event.pointerId)}
          onLostPointerCapture={(event) => finishDrawing(event.pointerId)}
        />
      </div>
    </section>
  );
}
