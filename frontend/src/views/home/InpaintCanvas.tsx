import { useEffect, useRef, useState, type PointerEvent } from 'react';
import { Eraser, RotateCcw } from 'lucide-react';

interface InpaintCanvasProps {
  image: string;
  brushSize: number;
  onBrushSizeChange: (brushSize: number) => void;
  onMaskChange: (mask: string | null) => void;
}

const MASK_SIZE = 768;

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
  const [eraser, setEraser] = useState(false);
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
    onMaskChange(null);
  }, [canvasSize, onMaskChange]);

  function drawTo(point: { x: number; y: number }) {
    const canvas = canvasRef.current;
    const context = canvas?.getContext('2d');
    const lastPoint = lastPointRef.current;
    if (!canvas || !context || !lastPoint) return;

    context.save();
    context.globalCompositeOperation = eraser ? 'destination-out' : 'source-over';
    context.strokeStyle = 'rgba(255,255,255,0.9)';
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

  function finishDrawing() {
    const canvas = canvasRef.current;
    drawingRef.current = false;
    lastPointRef.current = null;
    onMaskChange(canvas ? exportMask(canvas) : null);
  }

  function clearMask() {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
    onMaskChange(null);
  }

  return (
    <section className="inpaint-editor">
      <div className="inpaint-toolbar">
        <label className="field">
          <span>Brush {brushSize}px</span>
          <input type="range" min={8} max={160} step={2} value={brushSize} onChange={(event) => onBrushSizeChange(Number(event.currentTarget.value))} />
        </label>
        <button className={eraser ? 'mini-button active' : 'mini-button'} type="button" onClick={() => setEraser((value) => !value)} title="Erase mask strokes">
          <Eraser size={14} />
        </button>
        <button className="mini-button" type="button" onClick={clearMask} title="Clear inpaint mask">
          <RotateCcw size={14} />
        </button>
      </div>
      <div className="inpaint-stage" style={{ aspectRatio: `${canvasSize.width} / ${canvasSize.height}` }}>
        <img src={image} alt="Inpaint reference" draggable={false} />
        <canvas
          ref={canvasRef}
          aria-label="Inpaint mask canvas"
          onPointerDown={(event) => {
            event.currentTarget.setPointerCapture(event.pointerId);
            drawingRef.current = true;
            lastPointRef.current = pointFromEvent(event, event.currentTarget);
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
