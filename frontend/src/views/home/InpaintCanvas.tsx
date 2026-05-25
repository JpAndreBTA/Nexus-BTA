import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { Brush, Eraser, Pencil, RotateCcw, RotateCw, Trash2 } from 'lucide-react';

interface InpaintCanvasProps {
  image: string;
  brushSize: number;
  onBrushSizeChange: (brushSize: number) => void;
  onMaskChange: (mask: string | null) => void;
}

const MASK_SIZE = 768;
type PaintTool = 'mask' | 'eraser' | 'sketch';

function pointFromEvent(event: PointerEvent<HTMLCanvasElement>, canvas: HTMLCanvasElement) {
  const rect = canvas.getBoundingClientRect();
  return {
    x: ((event.clientX - rect.left) / rect.width) * canvas.width,
    y: ((event.clientY - rect.top) / rect.height) * canvas.height,
  };
}

function exportMask(canvas: HTMLCanvasElement) {
  const source = canvas.getContext('2d');
  if (!source) return null;

  const pixels = source.getImageData(0, 0, canvas.width, canvas.height);
  let hasMask = false;
  for (let index = 0; index < pixels.data.length; index += 4) {
    const alpha = pixels.data[index + 3];
    if (alpha > 3) {
      pixels.data[index] = 255;
      pixels.data[index + 1] = 255;
      pixels.data[index + 2] = 255;
      pixels.data[index + 3] = 255;
      hasMask = true;
    } else {
      pixels.data[index] = 0;
      pixels.data[index + 1] = 0;
      pixels.data[index + 2] = 0;
      pixels.data[index + 3] = 255;
    }
  }

  if (!hasMask) return null;
  const out = document.createElement('canvas');
  out.width = canvas.width;
  out.height = canvas.height;
  out.getContext('2d')?.putImageData(pixels, 0, 0);
  return out.toDataURL('image/png');
}

export function InpaintCanvas({ image, brushSize, onBrushSizeChange, onMaskChange }: InpaintCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const drawingRef = useRef(false);
  const lastPointRef = useRef<{ x: number; y: number } | null>(null);
  const [tool, setTool] = useState<PaintTool>('mask');
  const [sketchColor, setSketchColor] = useState('#ff3b3b');
  const [undoStack, setUndoStack] = useState<string[]>([]);
  const [redoStack, setRedoStack] = useState<string[]>([]);
  const [canvasSize, setCanvasSize] = useState({ width: MASK_SIZE, height: MASK_SIZE });

  useEffect(() => {
    const source = new Image();
    source.onload = () => {
      const naturalWidth = source.naturalWidth || MASK_SIZE;
      const naturalHeight = source.naturalHeight || MASK_SIZE;
      const scale = MASK_SIZE / Math.max(naturalWidth, naturalHeight);
      setCanvasSize({
        width: Math.max(16, Math.round(naturalWidth * scale)),
        height: Math.max(16, Math.round(naturalHeight * scale)),
      });
    };
    source.src = image;
  }, [image]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width = canvasSize.width;
    canvas.height = canvasSize.height;
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    setUndoStack([]);
    setRedoStack([]);
    onMaskChange(null);
  }, [canvasSize, onMaskChange]);

  function snapshot() {
    const canvas = canvasRef.current;
    return canvas ? canvas.toDataURL('image/png') : '';
  }

  function restore(dataUrl: string) {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    if (!canvas || !context) return;
    context.clearRect(0, 0, canvas.width, canvas.height);
    if (!dataUrl) {
      onMaskChange(null);
      return;
    }
    const imageElement = new Image();
    imageElement.onload = () => {
      context.clearRect(0, 0, canvas.width, canvas.height);
      context.drawImage(imageElement, 0, 0, canvas.width, canvas.height);
      onMaskChange(exportMask(canvas));
    };
    imageElement.src = dataUrl;
  }

  function pushUndo() {
    setUndoStack((current) => [...current.slice(-24), snapshot()]);
    setRedoStack([]);
  }

  function drawTo(point: { x: number; y: number }) {
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

  function drawDot(point: { x: number; y: number }) {
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

  function finishDrawing() {
    const canvas = canvasRef.current;
    drawingRef.current = false;
    lastPointRef.current = null;
    onMaskChange(canvas ? exportMask(canvas) : null);
  }

  function clearMask() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    pushUndo();
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    onMaskChange(null);
  }

  function undo() {
    setUndoStack((current) => {
      if (!current.length) return current;
      const previous = current[current.length - 1];
      setRedoStack((redo) => [...redo.slice(-24), snapshot()]);
      restore(previous);
      return current.slice(0, -1);
    });
  }

  function redo() {
    setRedoStack((current) => {
      if (!current.length) return current;
      const next = current[current.length - 1];
      setUndoStack((undoHistory) => [...undoHistory.slice(-24), snapshot()]);
      restore(next);
      return current.slice(0, -1);
    });
  }

  return (
    <section className="inpaint-editor">
      <div className="inpaint-toolbar">
        <div className="tool-segment">
          <button className={tool === 'mask' ? 'active' : ''} type="button" onClick={() => setTool('mask')} title="Mask brush">
            <Brush size={14} />
          </button>
          <button className={tool === 'eraser' ? 'active' : ''} type="button" onClick={() => setTool('eraser')} title="Erase mask">
            <Eraser size={14} />
          </button>
          <button className={tool === 'sketch' ? 'active' : ''} type="button" onClick={() => setTool('sketch')} title="Inpaint sketch">
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
        <button className="mini-button" type="button" onClick={undo} disabled={!undoStack.length} title="Undo">
          <RotateCcw size={14} />
        </button>
        <button className="mini-button" type="button" onClick={redo} disabled={!redoStack.length} title="Redo">
          <RotateCw size={14} />
        </button>
        <button className="mini-button" type="button" onClick={clearMask} title="Clear inpaint mask">
          <Trash2 size={14} />
        </button>
      </div>
      <div className="inpaint-stage" style={{ aspectRatio: `${canvasSize.width} / ${canvasSize.height}` }}>
        <img src={image} alt="Inpaint reference" draggable={false} />
        <canvas
          ref={canvasRef}
          aria-label="Inpaint mask canvas"
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            pushUndo();
            drawingRef.current = true;
            const point = pointFromEvent(event, event.currentTarget);
            lastPointRef.current = point;
            drawDot(point);
          }}
          onPointerMove={(event) => {
            if (!drawingRef.current) return;
            drawTo(pointFromEvent(event, event.currentTarget));
          }}
          onPointerUp={finishDrawing}
          onPointerCancel={finishDrawing}
          onPointerLeave={() => {
            if (drawingRef.current) finishDrawing();
          }}
        />
      </div>
    </section>
  );
}
