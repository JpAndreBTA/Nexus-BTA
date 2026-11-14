import { useEffect, useRef, useState } from 'react';

import { fetchBoundedGlb, parseGlbPreview, type PreviewGeometry } from './glbPreview';

const vertexShader = "#version 300 es\nin vec3 a_position;\nin vec3 a_normal;\nuniform vec3 u_center;\nuniform float u_extent;\nuniform float u_yaw;\nuniform float u_pitch;\nuniform float u_zoom;\nuniform vec2 u_aspect;\nout float v_light;\nvec3 rotatePoint(vec3 p) {\n  float cy=cos(u_yaw), sy=sin(u_yaw);\n  vec3 y=vec3(cy*p.x+sy*p.z,p.y,-sy*p.x+cy*p.z);\n  float cp=cos(u_pitch), sp=sin(u_pitch);\n  return vec3(y.x,cp*y.y-sp*y.z,sp*y.y+cp*y.z);\n}\nvoid main() {\n  vec3 pos=rotatePoint((a_position-u_center)/max(u_extent,0.000001));\n  vec3 normal=normalize(rotatePoint(a_normal));\n  v_light=0.30 + 0.70*max(0.0,dot(normal,normalize(vec3(0.40,0.60,0.70))));\n  gl_Position=vec4(pos.x*u_aspect.x*u_zoom*1.45,\n                   pos.y*u_aspect.y*u_zoom*1.45,pos.z*0.70,1.0);\n}";
const fragmentShader = "#version 300 es\nprecision mediump float;\nin float v_light;\nout vec4 fragColor;\nvoid main(){fragColor=vec4(vec3(0.58,0.75,0.88)*v_light,1.0);}";

function compile(gl: WebGL2RenderingContext, type: number, source: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error('Unable to initialize graphics shader.');
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    gl.deleteShader(shader);
    throw new Error('3D preview shader compilation failed.');
  }
  return shader;
}

function createRenderer(canvas: HTMLCanvasElement, geometry: PreviewGeometry): () => void {
  const gl = canvas.getContext('webgl2', { alpha: false, antialias: true });
  if (!gl) throw new Error('WebGL2 is unavailable; download the GLB to inspect it in Blender.');
  const vs = compile(gl, gl.VERTEX_SHADER, vertexShader);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fragmentShader);
  const program = gl.createProgram();
  if (!program) throw new Error('3D preview graphics program unavailable.');
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);
  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    gl.deleteProgram(program);
    throw new Error('Unable to link GLB preview.');
  }

  const position = gl.createBuffer();
  const normal = gl.createBuffer();
  if (!position || !normal) {
    gl.deleteProgram(program);
    throw new Error('Unable to allocate GLB preview vertex buffers.');
  }
  gl.useProgram(program);
  const pIndex = gl.getAttribLocation(program, 'a_position');
  const nIndex = gl.getAttribLocation(program, 'a_normal');
  gl.bindBuffer(gl.ARRAY_BUFFER, position);
  gl.bufferData(gl.ARRAY_BUFFER, geometry.positions, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(pIndex);
  gl.vertexAttribPointer(pIndex, 3, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, normal);
  gl.bufferData(gl.ARRAY_BUFFER, geometry.normals, gl.STATIC_DRAW);
  gl.enableVertexAttribArray(nIndex);
  gl.vertexAttribPointer(nIndex, 3, gl.FLOAT, false, 0, 0);
  gl.uniform3fv(gl.getUniformLocation(program, 'u_center'), geometry.center);
  gl.uniform1f(gl.getUniformLocation(program, 'u_extent'), geometry.extent);
  const uYaw = gl.getUniformLocation(program, 'u_yaw');
  const uPitch = gl.getUniformLocation(program, 'u_pitch');
  const uZoom = gl.getUniformLocation(program, 'u_zoom');
  const uAspect = gl.getUniformLocation(program, 'u_aspect');
  let yaw = 0.5, pitch = -0.3, zoom = 1;
  let pointer: { id: number; x: number; y: number } | null = null;

  const draw = () => {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(canvas.clientWidth * dpr));
    const height = Math.max(1, Math.round(canvas.clientHeight * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0.045, 0.055, 0.075, 1);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
    gl.enable(gl.DEPTH_TEST);
    gl.useProgram(program);
    gl.uniform1f(uYaw, yaw);
    gl.uniform1f(uPitch, pitch);
    gl.uniform1f(uZoom, zoom);
    gl.uniform2f(uAspect, width >= height ? height / width : 1, height >= width ? width / height : 1);
    gl.drawArrays(gl.TRIANGLES, 0, geometry.triangles * 3);
  };
  const onDown = (event: PointerEvent) => {
    if (pointer || (event.pointerType === 'mouse' && event.button !== 0)) return;
    pointer = { id: event.pointerId, x: event.clientX, y: event.clientY };
    canvas.setPointerCapture(event.pointerId);
  };
  const onMove = (event: PointerEvent) => {
    if (!pointer || pointer.id !== event.pointerId) return;
    yaw += (event.clientX - pointer.x) * 0.009;
    pitch = Math.max(-1.46, Math.min(1.46, pitch + (event.clientY - pointer.y) * 0.009));
    pointer.x = event.clientX;
    pointer.y = event.clientY;
    draw();
  };
  const onUp = (event: PointerEvent) => {
    if (pointer?.id === event.pointerId) pointer = null;
  };
  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    zoom = Math.max(0.5, Math.min(3, zoom * (event.deltaY > 0 ? 0.9 : 1.1)));
    draw();
  };
  const onKey = (event: KeyboardEvent) => {
    if (event.key === 'ArrowLeft') yaw -= 0.1;
    else if (event.key === 'ArrowRight') yaw += 0.1;
    else if (event.key === 'ArrowUp') pitch = Math.max(-1.46, pitch - 0.1);
    else if (event.key === 'ArrowDown') pitch = Math.min(1.46, pitch + 0.1);
    else if (event.key === '+') zoom = Math.min(3, zoom * 1.1);
    else if (event.key === '-') zoom = Math.max(0.5, zoom / 1.1);
    else if (event.key === 'Home') { yaw = 0.5; pitch = -0.3; zoom = 1; }
    else return;
    event.preventDefault();
    draw();
  };
  canvas.addEventListener('pointerdown', onDown);
  canvas.addEventListener('pointermove', onMove);
  canvas.addEventListener('pointerup', onUp);
  canvas.addEventListener('pointercancel', onUp);
  canvas.addEventListener('lostpointercapture', onUp);
  canvas.addEventListener('wheel', onWheel, { passive: false });
  canvas.addEventListener('keydown', onKey);
  const observer = new ResizeObserver(draw);
  observer.observe(canvas);
  draw();
  return () => {
    observer.disconnect();
    canvas.removeEventListener('pointerdown', onDown);
    canvas.removeEventListener('pointermove', onMove);
    canvas.removeEventListener('pointerup', onUp);
    canvas.removeEventListener('pointercancel', onUp);
    canvas.removeEventListener('lostpointercapture', onUp);
    canvas.removeEventListener('wheel', onWheel);
    canvas.removeEventListener('keydown', onKey);
    gl.deleteBuffer(position);
    gl.deleteBuffer(normal);
    gl.deleteProgram(program);
  };
}

export function GalleryGlbPreview({ src, filename }: { src: string; filename: string }) {
  const canvas = useRef<HTMLCanvasElement | null>(null);
  const [message, setMessage] = useState('Loading limited GLB geometry preview…');
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const controller = new AbortController();
    let disposed = false;
    let release: (() => void) | undefined;
    setReady(false);
    setMessage('Loading limited GLB geometry preview…');
    async function load() {
      try {
        const file = await fetchBoundedGlb(src, controller.signal);
        const mesh = parseGlbPreview(file);
        if (disposed || !canvas.current) return;
        release = createRenderer(canvas.current, mesh);
        setReady(true);
        setMessage('First mesh primitive · ' + mesh.triangles.toLocaleString() + ' triangles · geometry only');
      } catch (error) {
        if (disposed || controller.signal.aborted) return;
        setMessage(error instanceof Error ? error.message : 'This GLB cannot be previewed.');
      }
    }
    void load();
    return () => {
      disposed = true;
      controller.abort();
      release?.();
    };
  }, [src]);
  return (
    <div className="gallery-glb-viewer">
      <canvas ref={canvas} tabIndex={0} role="img" aria-label={'Experimental GLB orbit preview of ' + filename} />
      <div className="gallery-glb-legend" role="status">
        <span>{message}</span>
        {ready && <small>Drag to orbit · wheel or +/− to zoom · arrow keys rotate · Home resets</small>}
        <small>Materials, scene transforms, animations and additional meshes are not rendered.</small>
        <a href={src} download={filename}>Download original GLB</a>
      </div>
    </div>
  );
}
