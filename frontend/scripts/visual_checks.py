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
    frame_dir.mkdir(exist_ok=True)
    for old in frame_dir.glob("*.png"):
        old.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=128:128",
            "-frames:v",
            str(frames),
            str(frame_dir / "frame_%03d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    decoded = sorted(frame_dir.glob("frame_*.png"))
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
        "stream": ffprobe_video(path),
        "frames_sampled": len(decoded),
        "avg_high_saturation_fraction": float(np.mean([item["high_saturation_fraction"] for item in image_metrics])),
        "avg_local_diff": float(np.mean([item["local_diff"] for item in image_metrics])),
        "avg_luma_edge": float(np.mean(edge_scores)),
        "repeat_fraction": float(repeat_fraction),
        "consecutive_mad": consecutive,
    }
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
    return metrics
