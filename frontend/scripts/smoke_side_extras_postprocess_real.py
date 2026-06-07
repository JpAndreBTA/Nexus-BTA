import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
SOURCE_VIDEO = ROOT / "output" / "video" / "20260531_202058_Wan_i2v_NEXUS_BTA_WAN22_LOOP_CYCLE_00001_.mp4"
ENGINE = os.environ.get("SIDE_EXTRAS_ENGINE", "nvidia_rtx")
INTERPOLATE = os.environ.get("SIDE_EXTRAS_INTERPOLATE") == "1"
TARGET_FPS = os.environ.get("SIDE_EXTRAS_TARGET_FPS", "24")
SCALE = os.environ.get("SIDE_EXTRAS_SCALE", "2x")
DENOISE = os.environ.get("SIDE_EXTRAS_DENOISE", "1") != "0"
DETAIL = os.environ.get("SIDE_EXTRAS_DETAIL", "1") != "0"
FACE = os.environ.get("SIDE_EXTRAS_FACE", "1") != "0"
TIMEOUT_SECONDS = int(float(os.environ.get("SIDE_EXTRAS_TIMEOUT_SECONDS", "600")))


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )


def ensure_source_video() -> None:
    if not SOURCE_VIDEO.exists():
        raise AssertionError(f"Missing requested Wan battery video: {SOURCE_VIDEO}")


def ffprobe(path: Path) -> dict[str, object]:
    proc = run(
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
        ]
    )
    if proc.returncode != 0:
        raise AssertionError(f"ffprobe failed for {path}: {proc.stderr}")
    return (json.loads(proc.stdout or "{}").get("streams") or [{}])[0]


def sample_frames(path: Path, name: str) -> list[Path]:
    frames_dir = RESULTS / f"frames_{name}"
    frames_dir.mkdir(exist_ok=True)
    for old in frames_dir.glob("*.png"):
        old.unlink()
    proc = run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(path),
            "-vf",
            "scale=192:100",
            "-frames:v",
            "12",
            str(frames_dir / "frame_%03d.png"),
        ]
    )
    if proc.returncode != 0:
        raise AssertionError(f"Could not decode frames from {path}: {proc.stderr}")
    frames = sorted(frames_dir.glob("frame_*.png"))
    if len(frames) < 6:
        raise AssertionError(f"Expected multiple decoded frames from {path}, got {len(frames)}")
    return frames


def visual_metrics(path: Path, name: str) -> dict[str, object]:
    frames = sample_frames(path, name)
    rgb = [np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32) for frame in frames]
    consecutive = [float(np.mean(np.abs(rgb[i + 1] - rgb[i]))) for i in range(len(rgb) - 1)]
    repeated = sum(1 for value in consecutive if value < 0.5) / max(1, len(consecutive))
    local_diff = [
        float((np.abs(np.diff(frame, axis=0)).mean() + np.abs(np.diff(frame, axis=1)).mean()) / 2)
        for frame in rgb
    ]
    luma_edges = []
    for frame_path in frames:
        gray = np.asarray(Image.open(frame_path).convert("L"), dtype=np.float32)
        luma_edges.append(float((np.abs(np.diff(gray, axis=0)).mean() + np.abs(np.diff(gray, axis=1)).mean()) / 2))
    metrics = {
        "stream": ffprobe(path),
        "frames_sampled": len(frames),
        "repeat_fraction": repeated,
        "consecutive_mad": consecutive,
        "avg_motion_mad": float(np.mean(consecutive)),
        "avg_local_diff": float(np.mean(local_diff)),
        "avg_luma_edge": float(np.mean(luma_edges)),
    }
    if metrics["repeat_fraction"] > 0.65:
        raise AssertionError(f"{name} appears frozen/repeated: {metrics!r}")
    if metrics["avg_local_diff"] > 58:
        raise AssertionError(f"{name} looks noise-heavy: {metrics!r}")
    if metrics["avg_luma_edge"] < 5.0:
        raise AssertionError(f"{name} looks over-smoothed or flat: {metrics!r}")
    return metrics


def make_comparison_sheet(source: Path, output: Path) -> Path:
    sheet = RESULTS / f"front_side_extras_postprocess_{ENGINE}_compare.jpg"
    proc = run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-i",
            str(output),
            "-filter_complex",
            "[0:v]select='not(mod(n,6))',scale=232:120,tile=4x1[src];"
            "[1:v]select='not(mod(n,6))',scale=232:120,tile=4x1[out];"
            "[src][out]vstack=inputs=2",
            "-frames:v",
            "1",
            str(sheet),
        ]
    )
    if proc.returncode != 0:
        raise AssertionError(f"Could not create comparison sheet: {proc.stderr}")
    return sheet


def metadata_for(output_path: Path) -> dict[str, object]:
    meta_path = output_path.with_name(output_path.name + ".nexus.json")
    if not meta_path.exists():
        raise AssertionError(f"Missing Extras metadata sidecar: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def output_path_from_url(url: str) -> Path:
    if "/outputs/" not in url:
        raise AssertionError(f"Unexpected output URL: {url}")
    rel = url.split("/outputs/", 1)[1].split("?", 1)[0]
    return ROOT / "output" / Path(rel.replace("/", "\\"))


def main() -> None:
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise AssertionError("ffmpeg and ffprobe are required for the visual side Extras battery.")
    ensure_source_video()
    source_metrics = visual_metrics(SOURCE_VIDEO, "side_extras_source_full")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.set_default_timeout(900000)
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(f"{BASE}?side_extras_smoke={int(time.time())}", wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof maybeRunPostGenerationExtras === 'function'", timeout=120000)
        page.evaluate(
            """
            async () => {
              const fps = document.querySelector('#fpsInput');
              if (fps) {
                fps.value = '16';
                fps.dispatchEvent(new Event('input', { bubbles: true }));
                fps.dispatchEvent(new Event('change', { bubbles: true }));
              }
              const extras = document.querySelector('#extrasPostEnableToggle');
              extras.checked = true;
              document.querySelector('#extrasPostOutputMode').value = 'video';
              document.querySelector('#extrasPostUpscaleEngine').value = '%s';
              document.querySelector('#extrasPostScale').value = '%s';
              document.querySelector('#extrasPostInterpolateToggle').checked = %s;
              document.querySelector('#extrasPostTargetFps').value = '%s';
              document.querySelector('#sideDenoiseToggle').checked = %s;
              document.querySelector('#sideDenoiseStrengthSlider').value = '0.24';
              document.querySelector('#sideDetailToggle').checked = %s;
              document.querySelector('#sideDetailStrengthSlider').value = '0.28';
              document.querySelector('#sideFaceRestoreToggle').checked = %s;
              syncSideExtrasPostUi({ prompt: false });
              syncSideRefineUi({ prompt: false });
              const ready = await ensureSideExtrasPostReady();
              if (!ready) throw new Error('Side Extras prerequisites were rejected.');
            }
            """ % (ENGINE, SCALE, "true" if INTERPOLATE else "false", TARGET_FPS, "true" if DENOISE else "false", "true" if DETAIL else "false", "true" if FACE else "false")
        )
        result = page.evaluate(
            """
            async ({ timeoutSeconds }) => {
              const response = {
                outputs: [{
                  kind: 'video',
                  media_type: 'video',
                  filename: '20260531_202058_Wan_i2v_NEXUS_BTA_WAN22_LOOP_CYCLE_00001_.mp4',
                  url: '/outputs/video/20260531_202058_Wan_i2v_NEXUS_BTA_WAN22_LOOP_CYCLE_00001_.mp4'
                }]
              };
              let timer = null;
              const run = maybeRunPostGenerationExtras(response, {
                prompt: 'front side Extras smoke test, clean upscale, low noise'
              });
              const timeout = new Promise((_, reject) => {
                timer = setTimeout(() => reject(new Error(`Side Extras battery timed out after ${timeoutSeconds}s`)), timeoutSeconds * 1000);
              });
              try {
                return await Promise.race([run, timeout]);
              } finally {
                if (timer) clearTimeout(timer);
              }
            }
            """,
            {"timeoutSeconds": TIMEOUT_SECONDS},
        )
        page.screenshot(path=str(RESULTS / "front_side_extras_postprocess_real.png"), full_page=True)
        browser.close()

    outputs = result.get("outputs") or []
    if not outputs:
        raise AssertionError(f"Side Extras did not return outputs: {result!r}")
    output_path = output_path_from_url(outputs[0]["url"])
    output_metrics = visual_metrics(output_path, "side_extras_final_output")
    stream = output_metrics["stream"]
    expected_factor = 4 if SCALE.startswith("4") else 2
    if int(stream.get("width") or 0) < 928 * expected_factor - 8 or int(stream.get("height") or 0) < 480 * expected_factor - 8:
        raise AssertionError(f"Expected 2x side Extras output dimensions, got {stream!r}")
    source_duration = float(source_metrics["stream"].get("duration") or 0)
    output_duration = float(output_metrics["stream"].get("duration") or 0)
    if source_duration and abs(output_duration - source_duration) > 0.25:
        raise AssertionError(f"Output duration drifted from full source: source={source_duration}, output={output_duration}")
    metadata = metadata_for(output_path)
    extras = metadata.get("extras", {})
    upscale = extras.get("upscale", {})
    denoise = extras.get("denoise", {})
    detail = extras.get("detail_refine", {})
    face = extras.get("face_restore", {})
    if ENGINE == "nvidia_rtx":
        if upscale.get("runtime_engine") != "nvidia_rtx":
            raise AssertionError(f"Expected real NVIDIA RTX runtime, got {upscale!r}")
        if upscale.get("workflow_reference") != "nvvfx.VideoSuperRes":
            raise AssertionError(f"Expected nvvfx.VideoSuperRes workflow reference, got {upscale!r}")
    elif ENGINE == "nvidia_pid":
        if upscale.get("runtime_engine") != "nvidia_pid" or "PiD" not in str(upscale.get("workflow_reference") or ""):
            raise AssertionError(f"Expected NVIDIA PiD staged runtime, got {upscale!r}")
    elif ENGINE in {"flashvsr", "seedvr2"}:
        if upscale.get("runtime_engine") != ENGINE:
            raise AssertionError(f"Expected real {ENGINE} runtime, got {upscale!r}")
        if upscale.get("fallback_reason"):
            raise AssertionError(f"Unexpected upscale fallback: {upscale!r}")
        expected_references = {"WavespeedFlashVSRNode", "AILab_FlashVSR_Advanced"} if ENGINE == "flashvsr" else {"SeedVR2VideoUpscaler"}
        if upscale.get("workflow_reference") not in expected_references:
            raise AssertionError(f"Expected one of {sorted(expected_references)}, got {upscale!r}")
    if DENOISE and (not denoise.get("enabled") or "atadenoise" not in str(denoise.get("runtime", ""))):
        raise AssertionError(f"Expected advanced denoise chain, got {denoise!r}")
    if DETAIL and (not detail.get("enabled") or detail.get("runtime") != "ffmpeg_unsharp"):
        raise AssertionError(f"Expected detail refine runtime, got {detail!r}")
    if FACE and not face.get("enabled"):
        raise AssertionError(f"Expected side Refine face restoration enabled, got {face!r}")
    sheet = make_comparison_sheet(SOURCE_VIDEO, output_path)
    report = {
        "source": str(SOURCE_VIDEO),
        "output": str(output_path),
        "comparison_sheet": str(sheet),
        "source_metrics": source_metrics,
        "output_metrics": output_metrics,
        "metadata": {
            "upscale": upscale,
            "denoise": denoise,
            "detail_refine": detail,
            "face_restore": face,
        },
        "front_response": result,
    }
    (RESULTS / f"front_side_extras_postprocess_{ENGINE}_real.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ok side Extras post-process battery: {output_path}")


if __name__ == "__main__":
    main()
