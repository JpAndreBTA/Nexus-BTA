import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)


def local_output_path(output: dict[str, object]) -> Path:
    path = str(output.get("path") or output.get("filename") or "")
    if not path:
        raise AssertionError(f"Output did not expose a path: {output!r}")
    return ROOT / "output" / Path(path.replace("\\", "/"))


def ffprobe_video(path: Path) -> dict[str, object]:
    if not shutil.which("ffprobe"):
        return {}
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,nb_frames,duration,r_frame_rate,bit_rate",
            "-of",
            "json",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError:
        return {}
    return (data.get("streams") or [{}])[0]


def _video_frame_count(stream: dict[str, object]) -> int:
    try:
        frame_count = int(float(stream.get("nb_frames") or 0))
    except (TypeError, ValueError):
        frame_count = 0
    if frame_count > 0:
        return frame_count
    try:
        duration = float(stream.get("duration") or 0)
        rate_text = str(stream.get("r_frame_rate") or "0/1")
        numerator, denominator = rate_text.split("/", 1)
        fps = float(numerator) / max(0.001, float(denominator))
        return int(round(duration * fps))
    except (TypeError, ValueError, ZeroDivisionError):
        return 0


def _extract_video_samples(path: Path, frame_dir: Path, frames: int, stream: dict[str, object], evenly: bool) -> list[Path]:
    frame_dir.mkdir(exist_ok=True)
    for old in frame_dir.glob("*.png"):
        old.unlink()
    command = ["ffmpeg", "-y", "-v", "error", "-i", str(path)]
    if evenly:
        frame_count = _video_frame_count(stream)
        sample_count = max(2, min(frames, frame_count or frames))
        if frame_count > 1:
            indexes = sorted({round(index * (frame_count - 1) / (sample_count - 1)) for index in range(sample_count)})
        else:
            indexes = list(range(sample_count))
        selector = "+".join(f"eq(n\\,{index})" for index in indexes)
        command.extend(["-vf", f"select='{selector}',scale=128:128", "-vsync", "0"])
    else:
        command.extend(["-vf", "scale=128:128", "-frames:v", str(frames)])
    command.append(str(frame_dir / "frame_%03d.png"))
    subprocess.run(command, cwd=ROOT, check=True)
    return sorted(frame_dir.glob("frame_*.png"))


def analyze_image(path: Path, case: str) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"{case}: missing generated image {path}")
    arr = np.asarray(Image.open(path).convert("RGB").resize((128, 128)), dtype=np.float32)
    channel_max = arr.max(axis=2)
    channel_min = arr.min(axis=2)
    sat = np.zeros_like(channel_max)
    np.divide(channel_max - channel_min, channel_max, out=sat, where=channel_max > 1)
    local_diff = float((np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()) / 2)
    metrics = {
        "path": str(path),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "dark_fraction": float((channel_max < 24).mean()),
        "bright_fraction": float((channel_min > 232).mean()),
        "high_saturation_fraction": float((sat > 0.65).mean()),
        "local_diff": local_diff,
    }
    if metrics["std"] < 2.0:
        raise AssertionError(f"{case}: image appears blank/flat: {metrics!r}")
    if metrics["high_saturation_fraction"] > 0.55 and metrics["local_diff"] > 35:
        raise AssertionError(f"{case}: image looks artifact/noise-heavy: {metrics!r}")
    return metrics


def analyze_video(path: Path, case: str, frames: int = 8, require_motion: bool = False) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"{case}: missing generated video {path}")
    if not shutil.which("ffmpeg"):
        raise AssertionError("ffmpeg is required for visual validation.")
    frame_dir = RESULTS / f"frames_{case}"
    stream = ffprobe_video(path)
    decoded = _extract_video_samples(path, frame_dir, frames, stream, evenly=require_motion)
    if len(decoded) < 2:
        raise AssertionError(f"{case}: too few decoded frames from {path}")
    arrays = [np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32) for frame in decoded]
    image_metrics = [analyze_image(frame, f"{case}_{frame.stem}") for frame in decoded]
    luma_arrays = [np.asarray(Image.open(frame).convert("L"), dtype=np.float32) for frame in decoded]
    edge_scores = [
        float((np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()) / 2)
        for arr in luma_arrays
    ]
    consecutive = [float(np.mean(np.abs(arrays[index + 1] - arrays[index]))) for index in range(len(arrays) - 1)]
    repeat_fraction = sum(1 for value in consecutive if value < 0.5) / max(1, len(consecutive))
    metrics = {
        "path": str(path),
        "stream": stream,
        "frames_sampled": len(decoded),
        "avg_high_saturation_fraction": float(np.mean([item["high_saturation_fraction"] for item in image_metrics])),
        "avg_local_diff": float(np.mean([item["local_diff"] for item in image_metrics])),
        "avg_luma_edge": float(np.mean(edge_scores)),
        "repeat_fraction": float(repeat_fraction),
        "consecutive_mad": consecutive,
    }
    interior = consecutive[1:-1] if len(consecutive) > 3 else consecutive
    metrics["interior_avg_motion_mad"] = float(np.mean(interior)) if interior else 0.0
    metrics["interior_static_fraction"] = float(sum(1 for value in interior if value < 1.25) / max(1, len(interior)))
    try:
        bit_rate = int(float(metrics["stream"].get("bit_rate") or 0))
    except (TypeError, ValueError):
        bit_rate = 0
    metrics["bit_rate"] = bit_rate
    if metrics["avg_high_saturation_fraction"] > 0.50 and metrics["avg_local_diff"] > 35:
        raise AssertionError(f"{case}: video looks artifact/noise-heavy: {metrics!r}")
    if bit_rate > 3_000_000 and metrics["avg_high_saturation_fraction"] > 0.30 and metrics["avg_local_diff"] > 20:
        raise AssertionError(f"{case}: video looks like repeated high-saturation latent/upscale artifact: {metrics!r}")
    if metrics["avg_luma_edge"] < 8.0 and metrics["avg_local_diff"] < 10.0:
        raise AssertionError(f"{case}: video looks over-blurred/identity-melted: {metrics!r}")
    if require_motion and repeat_fraction > 0.65:
        raise AssertionError(f"{case}: video appears frozen/repeated: {metrics!r}")
    if require_motion and metrics["interior_static_fraction"] > 0.65 and metrics["interior_avg_motion_mad"] < 1.75:
        raise AssertionError(f"{case}: video only moves at endpoint jumps; interior appears frozen: {metrics!r}")
    return metrics
