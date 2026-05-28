import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
VIDEO = ROOT / "input" / "nexus_ltx_wan_motion_guide_4813b9fc9a.mp4"


def ffprobe(path: Path) -> dict[str, object]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,pix_fmt,nb_frames,duration,r_frame_rate",
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
    data = json.loads(proc.stdout or "{}")
    return (data.get("streams") or [{}])[0]


def visual_metrics(path: Path) -> dict[str, object]:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise AssertionError("ffmpeg/ffprobe are required for visual validation.")
    frames_dir = RESULTS / "frames_extras_ltx_detailer"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", "scale=128:128", "-frames:v", "8", str(frames_dir / "frame_%03d.png")],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) < 3:
        raise AssertionError(f"Expected multiple decoded frames from {path}")
    rgb = [np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32) for frame in frames]
    means = np.array([frame.mean() for frame in rgb], dtype=np.float32)
    high_sat = []
    local_diff = []
    for frame in rgb:
        channel_max = frame.max(axis=2)
        channel_min = frame.min(axis=2)
        sat = np.zeros_like(channel_max)
        np.divide(channel_max - channel_min, channel_max, out=sat, where=channel_max > 1)
        high_sat.append(float((sat > 0.65).mean()))
        local_diff.append(float((np.abs(np.diff(frame, axis=0)).mean() + np.abs(np.diff(frame, axis=1)).mean()) / 2))
    consecutive = [float(np.mean(np.abs(rgb[i + 1] - rgb[i]))) for i in range(len(rgb) - 1)]
    repeated = sum(1 for value in consecutive if value < 0.5) / max(1, len(consecutive))
    metrics = {
        "stream": ffprobe(path),
        "frames_sampled": len(frames),
        "mean_min": float(means.min()),
        "mean_max": float(means.max()),
        "avg_high_saturation_fraction": float(np.mean(high_sat)),
        "avg_local_diff": float(np.mean(local_diff)),
        "repeat_fraction": repeated,
        "consecutive_mad": consecutive,
    }
    pix_fmt = str(metrics["stream"].get("pix_fmt") or "")
    if "a" in pix_fmt.lower():
        raise AssertionError(f"Expected non-alpha MP4 pixel format, got {pix_fmt}")
    if metrics["avg_high_saturation_fraction"] > 0.45 and metrics["avg_local_diff"] > 25:
        raise AssertionError(f"Extras LTX Detailer output looks artifact/noise-heavy: {metrics!r}")
    if repeated > 0.65:
        raise AssertionError(f"Extras LTX Detailer output appears mostly frozen/repeated: {metrics!r}")
    return metrics


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof startExtrasProcessing === 'function'", timeout=120000)
        page.evaluate("() => { switchActivity('extras', document.querySelector('[data-activity=\"extras\"]')); setExtrasMode('video', { quiet: true }); }")
        page.locator("#extrasFileInput").set_input_files(str(VIDEO))
        page.wait_for_function("() => typeof extrasSource !== 'undefined' && !!extrasSource && extrasSource.mediaType === 'video'", timeout=60000)
        page.evaluate(
            """
            () => {
              document.querySelector('#extrasVideoUpscaleToggle').checked = true;
              document.querySelector('#extrasVideoUpscaleEngine').value = 'ltx_detailer';
              document.querySelector('#extrasVideoUpscaleFactor').value = '2x';
              document.querySelector('#extrasVideoInterpolateToggle').checked = true;
              document.querySelector('#extrasTargetFps').value = '16';
              document.querySelector('#extrasVideoAlphaToggle').checked = false;
              document.querySelector('#extrasVideoEncodeToggle').checked = true;
              const encoder = document.querySelector('#extrasVideoEncoder');
              if (encoder) encoder.value = 'mp4_h264';
              syncExtrasVideoUpscaleUi();
              syncExtrasInterpolateUi();
              syncExtrasAlphaUi();
              syncExtrasEncodeUi();
            }
            """
        )
        page.locator("#extrasProcessBtn").click()
        page.wait_for_function("() => typeof extrasProcessing !== 'undefined' && extrasProcessing === false && !!extrasProcessedUrl", timeout=240000)
        output_url = page.evaluate("() => extrasProcessedUrl")
        page.screenshot(path=str(RESULTS / "front_extras_ltx_detailer_real.png"), full_page=True)
        browser.close()

    rel = output_url.split("/outputs/", 1)[1].split("?", 1)[0]
    output_path = ROOT / "output" / rel.replace("/", "\\")
    metrics = visual_metrics(output_path)
    result = {"output": str(output_path), "url": output_url, "metrics": metrics}
    (RESULTS / "front_extras_ltx_detailer_real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ok extras LTX detailer real battery: {output_path}")


if __name__ == "__main__":
    main()
