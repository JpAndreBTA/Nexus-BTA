import json
import os
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
VIDEO = ROOT / "output" / "video" / "20260531_202058_Wan_i2v_NEXUS_BTA_WAN22_LOOP_CYCLE_00001_.mp4"
REFINE = os.environ.get("EXTRAS_REFINE") == "1"
FACE = os.environ.get("EXTRAS_FACE") == "1"
SUFFIX = "_refine" if REFINE else "_face" if FACE else ""


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
    frames_dir = RESULTS / f"frames_extras_nvidia_rtx{SUFFIX}"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(path), "-vf", "scale=192:100", "-frames:v", "12", str(frames_dir / "frame_%03d.png")],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) < 6:
        raise AssertionError(f"Expected multiple decoded frames from {path}")
    rgb = [np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32) for frame in frames]
    consecutive = [float(np.mean(np.abs(rgb[i + 1] - rgb[i]))) for i in range(len(rgb) - 1)]
    repeated = sum(1 for value in consecutive if value < 0.5) / max(1, len(consecutive))
    local_diff = [
        float((np.abs(np.diff(frame, axis=0)).mean() + np.abs(np.diff(frame, axis=1)).mean()) / 2)
        for frame in rgb
    ]
    means = np.array([frame.mean() for frame in rgb], dtype=np.float32)
    stream = ffprobe(path)
    if int(stream.get("width") or 0) < 1800 or int(stream.get("height") or 0) < 900:
        raise AssertionError(f"Expected 2x video dimensions, got {stream}")
    if repeated > 0.65:
        raise AssertionError(f"NVIDIA RTX Extras output appears mostly frozen/repeated: {consecutive}")
    if float(np.mean(local_diff)) > 55:
        raise AssertionError(f"NVIDIA RTX Extras output looks noise-heavy: local diff {float(np.mean(local_diff)):.2f}")
    return {
        "stream": stream,
        "frames_sampled": len(frames),
        "repeat_fraction": repeated,
        "consecutive_mad": consecutive,
        "avg_local_diff": float(np.mean(local_diff)),
        "mean_min": float(means.min()),
        "mean_max": float(means.max()),
    }


def make_contact_sheet(path: Path) -> Path:
    sheet = RESULTS / f"front_extras_nvidia_rtx_real{SUFFIX}_strip.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "select='not(mod(n,12))',scale=232:120,tile=5x2",
            "-frames:v",
            "1",
            str(sheet),
        ],
        cwd=ROOT,
        check=True,
    )
    return sheet


def main() -> None:
    if not VIDEO.exists():
        raise AssertionError(f"Missing requested battery video: {VIDEO}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof startExtrasProcessing === 'function'", timeout=120000)
        page.evaluate("() => { switchActivity('extras', document.querySelector('[data-activity=\"extras\"]')); setExtrasMode('video', { quiet: true }); }")
        page.locator("#extrasFileInput").set_input_files(str(VIDEO))
        page.wait_for_function("() => typeof extrasSource !== 'undefined' && !!extrasSource && extrasSource.mediaType === 'video'", timeout=60000)
        page.evaluate(
            """
            async () => {
              document.querySelector('#extrasVideoUpscaleToggle').checked = true;
              document.querySelector('#extrasVideoUpscaleEngine').value = 'nvidia_rtx';
              document.querySelector('#extrasVideoUpscaleFactor').value = '2x';
              document.querySelector('#extrasVideoInterpolateToggle').checked = false;
              document.querySelector('#extrasVideoDenoiseToggle').checked = %s;
              document.querySelector('#extrasVideoDetailRefineToggle').checked = %s;
              document.querySelector('#extrasVideoFaceRestoreToggle').checked = %s;
              document.querySelector('#extrasVideoDenoiseSlider').value = '0.18';
              document.querySelector('#extrasVideoDetailSlider').value = '0.28';
              document.querySelector('#extrasVideoAlphaToggle').checked = false;
              document.querySelector('#extrasVideoEncodeToggle').checked = true;
              const encoder = document.querySelector('#extrasVideoEncoder');
              if (encoder) encoder.value = 'mp4_h264';
              syncExtrasVideoUpscaleUi();
              syncExtrasVideoDenoiseUi();
              syncExtrasFaceRestoreUi('video', { promptDownload: false });
              if (%s && typeof maybeDownloadExtrasFaceRestoreModel === 'function') {
                await maybeDownloadExtrasFaceRestoreModel('extrasVideoFaceRestoreModel');
              }
              syncExtrasInterpolateUi();
              syncExtrasAlphaUi();
              syncExtrasEncodeUi();
            }
            """ % ("true" if REFINE else "false", "true" if REFINE else "false", "true" if FACE else "false", "true" if FACE else "false")
        )
        page.locator("#extrasProcessBtn").click()
        page.wait_for_function("() => typeof extrasProcessing !== 'undefined' && extrasProcessing === false && !!extrasProcessedUrl", timeout=600000)
        output_url = page.evaluate("() => extrasProcessedUrl")
        page.screenshot(path=str(RESULTS / f"front_extras_nvidia_rtx_real{SUFFIX}.png"), full_page=True)
        browser.close()

    rel = output_url.split("/outputs/", 1)[1].split("?", 1)[0]
    output_path = ROOT / "output" / rel.replace("/", "\\")
    metrics = visual_metrics(output_path)
    sheet = make_contact_sheet(output_path)
    result = {"output": str(output_path), "url": output_url, "contact_sheet": str(sheet), "metrics": metrics}
    (RESULTS / f"front_extras_nvidia_rtx_real{SUFFIX}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ok extras NVIDIA RTX real battery: {output_path}")


if __name__ == "__main__":
    main()
