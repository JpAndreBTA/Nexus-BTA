import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright


ROOT = Path(__file__).resolve().parents[2]


RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
MOTION_GUIDE = ROOT / "input" / "nexus_ltx_wan_motion_guide_56cae91f3b.mp4"
CAMERA_MOTION_GUIDE = ROOT / "input" / "CameraMan_ref.mp4"
TARGET_IMAGE = ROOT / "input" / "Smoke_Character.png"
START_REFERENCE = ROOT / "input" / "nexus_smoke_reference.png"
END_REFERENCE = ROOT / "input" / "nexus_smoke_reference2.jpg"
_modes_env = os.environ.get("NEXUS_LTX_MOTION_MODES")
MODES = tuple(
    item.strip().lower()
    for item in ((_modes_env if _modes_env is not None else "pose,canny,depth,camera").split(","))
    if item.strip()
)
WIDTH = int(os.environ.get("NEXUS_LTX_WIDTH") or "512")
HEIGHT = int(os.environ.get("NEXUS_LTX_HEIGHT") or "512")
STEPS = int(os.environ.get("NEXUS_LTX_STEPS") or "4")
CFG = float(os.environ.get("NEXUS_LTX_CFG") or "1")
FPS = float(os.environ.get("NEXUS_LTX_FPS") or "24")
SECONDS = float(os.environ.get("NEXUS_LTX_SECONDS") or "2")
ENABLE_DETAILER = os.environ.get("NEXUS_LTX_ENABLE_DETAILER", "").strip().lower() in {"1", "true", "yes", "on"}
RUN_MOTION = os.environ.get("NEXUS_LTX_RUN_MOTION", "1").strip().lower() not in {"0", "false", "no", "off"}
RUN_START_END = os.environ.get("NEXUS_LTX_RUN_START_END", "1").strip().lower() not in {"0", "false", "no", "off"}
STRICT_METRICS = os.environ.get("NEXUS_LTX_STRICT_METRICS", "0").strip().lower() in {"1", "true", "yes", "on", "strict"}
VIDEO_ARTIFACT_HIGHSAT_LIMIT = 0.32
VIDEO_ARTIFACT_SPREAD_LIMIT = 50.0
VIDEO_ARTIFACT_DIFF_LIMIT = 20.0
VIDEO_ARTIFACT_BITRATE_LIMIT = 3_000_000
MOTION_MIN_CHANGED_ANY10 = 0.35
MOTION_MIN_CHANGED_ANY15 = 0.18
MOTION_MIN_CONSECUTIVE_MAD = 3.0
MOTION_MIN_P95_MAX_DELTA = 24.0
START_END_MAX_REPEAT_FRACTION = 0.25
START_END_MAX_SPIKE_RATIO = 4.75
START_END_MIN_CHANGED_ANY10 = 0.12


def require_inputs() -> None:
    missing = [path for path in (MOTION_GUIDE, CAMERA_MOTION_GUIDE, TARGET_IMAGE, START_REFERENCE, END_REFERENCE) if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing requested frontend smoke input(s): " + ", ".join(str(path) for path in missing))


def motion_guide_for_mode(mode: str) -> Path:
    return CAMERA_MOTION_GUIDE if str(mode or "").strip().lower() == "camera" else MOTION_GUIDE


def js_json(page: Page, expression: str):
    return json.loads(page.evaluate(f"async () => JSON.stringify(await ({expression}))"))


def backend_fetch(page: Page, path: str):
    return js_json(page, f"nexusFetch('{path}')")


def wait_front_backend_sync(page: Page) -> dict[str, object]:
    page.goto(BASE, wait_until="networkidle", timeout=60000)
    page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
    page.wait_for_function(
        """() => backendOnline === true
          && Array.isArray(backendWorkflows)
          && backendCatalog?.categories
          && typeof nexusFetch === 'function'""",
        timeout=120000,
    )
    health = backend_fetch(page, "/health")
    if health.get("nexus") != "ok":
        raise AssertionError(f"Backend health did not sync through frontend: {health!r}")
    return health


def set_ltx_linear_defaults(page: Page) -> None:
    page.locator("button[data-preset='LTX']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'LTX' && currentActivity === 'img2img'")
    page.locator("#widthInput").fill(str(WIDTH))
    page.locator("#heightInput").fill(str(HEIGHT))
    page.locator("#stepsValue").fill(str(STEPS))
    page.locator("#cfgValue").fill(str(CFG))
    page.locator("#secondsInput").fill(str(SECONDS))
    page.locator("#fpsInput").fill(str(FPS))
    page.evaluate(
        """() => {
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          if (document.querySelector('#latentUpscaleSelect')) document.querySelector('#latentUpscaleSelect').value = 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors';
          if (document.querySelector('#activeAudioToggle')) document.querySelector('#activeAudioToggle').checked = false;
          syncGenerationActionUi();
        }"""
    )


def force_battery_fields(page: Page) -> None:
    page.locator("#widthInput").fill(str(WIDTH))
    page.locator("#heightInput").fill(str(HEIGHT))
    page.locator("#stepsValue").fill(str(STEPS))
    page.locator("#cfgValue").fill(str(CFG))
    page.locator("#fpsInput").fill(str(FPS))
    if not page.locator("#secondsInput").evaluate("el => !!el.readOnly"):
        page.locator("#secondsInput").fill(str(SECONDS))
    page.evaluate(
        """() => {
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          if (typeof syncLtxLinearMotionTransferTimeLock === 'function') syncLtxLinearMotionTransferTimeLock();
          if (document.querySelector('#distilledLoraOneSelect')) document.querySelector('#distilledLoraOneSelect').value = 'None';
          if (document.querySelector('#distilledLoraTwoSelect')) document.querySelector('#distilledLoraTwoSelect').value = 'ltx\\\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors';
          if (document.querySelector('#distilledLoraTwoStrength')) document.querySelector('#distilledLoraTwoStrength').value = '0.5';
          syncGenerationActionUi();
        }"""
    )


def enable_ltx_detailer(page: Page) -> None:
    if not ENABLE_DETAILER:
        return
    page.evaluate(
        """async () => {
          const toggle = document.querySelector('#ltxDetailerToggle');
          if (!toggle) throw new Error('LTX detailer toggle not found');
          toggle.checked = true;
          if (typeof maybeDownloadLtxDetailer === 'function') {
            const ok = await maybeDownloadLtxDetailer();
            if (!ok) throw new Error('LTX detailer model was not installed/selected');
          }
          const select = document.querySelector('#ltxDetailerLoraSelect');
          const concrete = [...(select?.options || [])].find(option => {
            const value = String(option.value || option.text || '').trim();
            return value && !/^(none|off|disabled|automatic|auto)$/i.test(value);
          });
          if (concrete) select.value = concrete.value;
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
    )
    payload = collect_payload(page)
    video = payload.get("video") if isinstance(payload, dict) else {}
    if not isinstance(video, dict) or video.get("detailer_enabled") is not True:
        raise AssertionError(f"Detailer toggle did not reach frontend payload: {payload!r}")
    detailer_lora = str(video.get("detailer_lora") or "")
    if not detailer_lora or detailer_lora.lower() in {"none", "automatic", "auto", "off"}:
        raise AssertionError(f"Detailer did not select a concrete model: {video!r}")


def collect_payload(page: Page) -> dict[str, object]:
    return js_json(page, "collectGenerationPayload()")


def assert_common_payload(payload: dict[str, object], case: str) -> None:
    if payload.get("preset") != "LTX" or payload.get("activity") != "img2img":
        raise AssertionError(f"{case}: expected LTX img2img payload, got {payload!r}")
    if payload.get("width") != WIDTH or payload.get("height") != HEIGHT:
        raise AssertionError(f"{case}: expected {WIDTH}x{HEIGHT}, got {payload.get('width')}x{payload.get('height')}")
    if payload.get("steps") != STEPS or payload.get("cfg") != CFG:
        raise AssertionError(f"{case}: expected {STEPS} steps cfg {CFG}, got steps={payload.get('steps')} cfg={payload.get('cfg')}")
    if ENABLE_DETAILER:
        video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
        if not isinstance(video, dict) or video.get("detailer_enabled") is not True:
            raise AssertionError(f"{case}: IC detailer is not enabled in frontend payload: {payload!r}")
        detailer_lora = str(video.get("detailer_lora") or "")
        if not detailer_lora or detailer_lora.lower() in {"none", "automatic", "auto", "off"}:
            raise AssertionError(f"{case}: IC detailer did not select a concrete model: {video!r}")


def poll_job(page: Page, job_id: str, timeout_ms: int = 900000) -> dict[str, object]:
    deadline = time.monotonic() + timeout_ms / 1000
    job: dict[str, object] = {}
    while time.monotonic() < deadline:
        job = backend_fetch(page, f"/generate/{job_id}")
        if job.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(2.5)
    else:
        raise TimeoutError(f"Job {job_id} did not finish within {timeout_ms}ms. Last status: {job!r}")
    if job.get("status") != "completed":
        raise AssertionError(f"Job {job_id} did not complete: {job!r}")
    if not job.get("outputs"):
        raise AssertionError(f"Job {job_id} completed without outputs: {job!r}")
    return job


def local_output_path(output: dict[str, object]) -> Path:
    path = str(output.get("path") or output.get("filename") or "")
    if not path:
        raise AssertionError(f"Output did not expose a path: {output!r}")
    return ROOT / "output" / Path(path.replace("\\", "/"))


def final_video_output(job: dict[str, object], case: str) -> dict[str, object]:
    outputs = job.get("outputs") or []
    if not isinstance(outputs, list):
        raise AssertionError(f"{case}: job outputs were not a list: {job!r}")
    candidates = [
        output
        for output in outputs
        if isinstance(output, dict)
        and str(output.get("path") or output.get("filename") or "").lower().endswith((".mp4", ".webm", ".mov"))
        and "motion_preprocess" not in str(output.get("path") or output.get("filename") or "").lower()
    ]
    if not candidates:
        raise AssertionError(f"{case}: no final generated video in job outputs: {outputs!r}")
    return candidates[0]


def video_bitrate(path: Path) -> int:
    if not shutil.which("ffprobe"):
        return 0
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=bit_rate",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    try:
        return int(float((proc.stdout or "0").strip() or "0"))
    except ValueError:
        return 0


def video_stream_info(path: Path) -> dict[str, object]:
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
            "stream=width,height,r_frame_rate,nb_frames,duration",
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
    streams = data.get("streams") or []
    return streams[0] if streams else {}


def analyze_video_visual_quality(path: Path, case: str) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"{case}: expected generated video at {path}")
    if not shutil.which("ffmpeg"):
        raise AssertionError("ffmpeg is required for frontend visual artifact validation.")
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
            "5",
            str(frame_dir / "frame_%03d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise AssertionError(f"{case}: ffmpeg did not extract frames from {path}")
    stats = []
    for frame in frames:
        arr = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32)
        channel_max = arr.max(axis=2)
        channel_min = arr.min(axis=2)
        saturation = np.zeros_like(channel_max, dtype=np.float32)
        np.divide(channel_max - channel_min, channel_max, out=saturation, where=channel_max > 1.0)
        local_diff = (
            np.abs(np.diff(arr, axis=0)).mean()
            + np.abs(np.diff(arr, axis=1)).mean()
        ) / 2.0
        stats.append(
            {
                "frame": frame.name,
                "dark_fraction": float((channel_max < 30).mean()),
                "high_saturation_fraction": float((saturation > 0.55).mean()),
                "channel_spread": float((channel_max - channel_min).mean()),
                "local_diff": float(local_diff),
                "mean": float(arr.mean()),
            }
        )
    avg = {
        "high_saturation_fraction": float(np.mean([item["high_saturation_fraction"] for item in stats])),
        "channel_spread": float(np.mean([item["channel_spread"] for item in stats])),
        "local_diff": float(np.mean([item["local_diff"] for item in stats])),
        "dark_fraction": float(np.mean([item["dark_fraction"] for item in stats])),
        "bit_rate": video_bitrate(path),
    }
    latent_noise_like = (
        avg["high_saturation_fraction"] > VIDEO_ARTIFACT_HIGHSAT_LIMIT
        and avg["channel_spread"] > VIDEO_ARTIFACT_SPREAD_LIMIT
        and avg["local_diff"] > VIDEO_ARTIFACT_DIFF_LIMIT
    )
    high_entropy_mp4 = avg["bit_rate"] > VIDEO_ARTIFACT_BITRATE_LIMIT and avg["high_saturation_fraction"] > 0.30
    if latent_noise_like or high_entropy_mp4:
        raise AssertionError(f"{case}: generated video looks like latent/color-noise artifact: {avg!r} at {path}")
    return {"path": str(path), "average": avg, "frames": stats}


def read_luma_frames(video_path: Path, case: str, prefix: str) -> np.ndarray:
    frame_dir = RESULTS / f"motion_frames_{case}_{prefix}"
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
            str(video_path),
            "-vf",
            "fps=8,scale=128:128",
            "-frames:v",
            "8",
            str(frame_dir / "frame_%03d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = []
    for frame in sorted(frame_dir.glob("frame_*.png")):
        rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32)
        frames.append(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    if len(frames) < 3:
        raise AssertionError(f"{case}: expected at least 3 sampled frames from {video_path}")
    return np.stack(frames)


def read_all_luma_frames(video_path: Path, case: str, prefix: str, max_frames: int = 65) -> np.ndarray:
    frame_dir = RESULTS / f"all_frames_{case}_{prefix}"
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
            str(video_path),
            "-vf",
            "scale=128:128",
            "-frames:v",
            str(max_frames),
            str(frame_dir / "frame_%03d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = []
    for frame in sorted(frame_dir.glob("frame_*.png")):
        rgb = np.asarray(Image.open(frame).convert("RGB"), dtype=np.float32)
        frames.append(0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2])
    if len(frames) < 5:
        raise AssertionError(f"{case}: expected at least 5 frames from {video_path}")
    return np.stack(frames)


def motion_stats(frames: np.ndarray) -> dict[str, float]:
    consecutive = np.abs(np.diff(frames, axis=0))
    from_start = np.abs(frames - frames[0])
    max_from_start = from_start.max(axis=0)
    return {
        "consecutive_mad": float(consecutive.mean()),
        "last_mad": float(from_start[-1].mean()),
        "max_mad": float(from_start.mean(axis=(1, 2)).max()),
        "changed_any10": float((max_from_start > 10).mean()),
        "changed_any15": float((max_from_start > 15).mean()),
        "p95_max_delta": float(np.percentile(max_from_start, 95)),
    }


def analyze_ltx_motion_transfer_motion(path: Path, case: str, guide_path: Path | None = None) -> dict[str, object]:
    guide_stats = motion_stats(read_luma_frames(guide_path or MOTION_GUIDE, case, "guide"))
    stats = motion_stats(read_luma_frames(path, case, "final"))
    is_camera = "camera" in case.lower()
    min_last_mad = max(8.0 if is_camera else 10.0, guide_stats["last_mad"] * (0.40 if is_camera else 0.45))
    min_changed_any10 = max(0.28 if is_camera else MOTION_MIN_CHANGED_ANY10, guide_stats["changed_any10"] * (0.50 if is_camera else 0.55))
    min_consecutive_mad = max(2.3 if is_camera else MOTION_MIN_CONSECUTIVE_MAD, guide_stats["consecutive_mad"] * (0.55 if is_camera else 0.65))
    frozen_like = (
        stats["last_mad"] < min_last_mad
        or stats["changed_any10"] < min_changed_any10
        or stats["consecutive_mad"] < min_consecutive_mad
    )
    control_only_like = (
        stats["changed_any15"] < MOTION_MIN_CHANGED_ANY15
        or stats["p95_max_delta"] < MOTION_MIN_P95_MAX_DELTA
    )
    if frozen_like or control_only_like:
        warning = (
            f"{case}: final video looks frozen or control-signal-only. "
            f"stats={stats!r}, guide={guide_stats!r}, path={path}"
        )
        if STRICT_METRICS:
            raise AssertionError(warning)
        return {"guide": guide_stats, "video": stats, "warning": warning}
    return {"guide": guide_stats, "video": stats}


def analyze_start_end_continuity(path: Path, payload: dict[str, object], case: str = "start_end") -> dict[str, object]:
    frames = read_all_luma_frames(path, case, "final")
    consecutive = np.abs(np.diff(frames, axis=0)).mean(axis=(1, 2))
    from_start = np.abs(frames - frames[0])
    max_from_start = from_start.max(axis=0)
    median_delta = float(np.median(consecutive))
    p95_delta = float(np.percentile(consecutive, 95))
    stats = {
        "frame_count_sampled": int(frames.shape[0]),
        "consecutive_mad_mean": float(consecutive.mean()),
        "consecutive_mad_median": median_delta,
        "consecutive_mad_p95": p95_delta,
        "consecutive_mad_max": float(consecutive.max()),
        "repeat_fraction": float((consecutive < 0.65).mean()),
        "spike_ratio": float(p95_delta / max(1.0, median_delta)),
        "changed_any10": float((max_from_start > 10).mean()),
        "last_mad": float(from_start[-1].mean()),
    }
    if stats["repeat_fraction"] > START_END_MAX_REPEAT_FRACTION:
        raise AssertionError(f"{case}: repeated/held frames suggest jumpy motion: {stats!r} at {path}")
    if stats["spike_ratio"] > START_END_MAX_SPIKE_RATIO and stats["consecutive_mad_p95"] > 14.0:
        raise AssertionError(f"{case}: frame-to-frame spikes suggest flicker: {stats!r} at {path}")
    if stats["changed_any10"] < START_END_MIN_CHANGED_ANY10 or stats["last_mad"] < 5.0:
        raise AssertionError(f"{case}: transition did not build continuous motion: {stats!r} at {path}")
    video = payload.get("video") if isinstance(payload.get("video"), dict) else {}
    expected_fps = max(24, FPS)
    expected_frames = int(round(SECONDS * expected_fps)) + 1
    if (expected_frames - 1) % 8 != 0:
        expected_frames = (((expected_frames - 1) // 8) + 1) * 8 + 1
    if int(round(float(video.get("fps") or 0))) != int(expected_fps) or int(video.get("frames") or 0) != expected_frames:
        raise AssertionError(f"{case}: frontend payload is not synced for smooth LTX transition fps/frames: {video!r}")
    info = video_stream_info(path)
    actual_frames = int(float(info.get("nb_frames") or 0))
    if actual_frames != expected_frames:
        raise AssertionError(f"{case}: output frame count drifted from payload: expected={expected_frames} actual={actual_frames} info={info!r}")
    if str(info.get("r_frame_rate") or "") not in {f"{int(expected_fps)}/1", f"{expected_fps}/1"}:
        raise AssertionError(f"{case}: output fps drifted from payload: expected={expected_fps} info={info!r}")
    stats["stream"] = info
    return stats


def latest_video_metadata(page: Page) -> dict[str, object]:
    gallery = backend_fetch(page, "/gallery")
    if not isinstance(gallery, list):
        raise AssertionError(f"Gallery did not return a list: {gallery!r}")
    videos = [
        item
        for item in gallery
        if str(item.get("filename") or item.get("path") or "").lower().endswith((".mp4", ".webm", ".mov"))
    ]
    if not videos:
        raise AssertionError("No video outputs found in backend gallery after smoke.")
    videos.sort(key=lambda item: item.get("modified") or item.get("mtime") or item.get("filename") or "", reverse=True)
    return videos[0]


def click_global_generate(page: Page) -> str:
    with page.expect_request("**/api/generate/start"):
        page.locator("#globalGenerateButton").click()
        page.wait_for_function("() => activeGenerationJobId", timeout=30000)
    return str(page.evaluate("() => activeGenerationJobId"))


def run_motion_transfer_mode(page: Page, mode: str) -> dict[str, object]:
    guide_path = motion_guide_for_mode(mode)
    page.evaluate("() => clearReferenceImage({ quiet: true })")
    page.locator("#referenceImageInput").set_input_files([str(guide_path), str(TARGET_IMAGE)])
    force_battery_fields(page)
    page.evaluate("mode => setLtxMotionTransferMode(mode)", mode)
    payload = collect_payload(page)
    assert_common_payload(payload, f"motion-{mode}-preprocess")
    video = payload.get("video")
    img2img = payload.get("img2img")
    if not isinstance(video, dict) or video.get("motion_transfer_enabled") is not True:
        raise AssertionError(f"motion-{mode}: preprocess payload did not enable motion transfer: {payload!r}")
    if video.get("motion_transfer_control_mode") != mode:
        raise AssertionError(f"motion-{mode}: expected mode {mode}, got {video.get('motion_transfer_control_mode')!r}")
    if not isinstance(img2img, dict) or not str(img2img.get("base_video") or "").startswith("data:video/"):
        raise AssertionError(f"motion-{mode}: missing video guide data URL.")
    refs = img2img.get("reference_images")
    if not isinstance(refs, list) or len(refs) != 1 or not str(refs[0]).startswith("data:image/"):
        raise AssertionError(f"motion-{mode}: missing Smoke_Character target image.")

    preprocess_item: dict[str, object] | None = None
    if mode != "camera":
        page.wait_for_function("() => shouldPreprocessLtxMotionTransfer()", timeout=60000)
        page.locator("#globalGenerateButton").click()
        page.wait_for_function("() => document.querySelector('#ltxMotionPreprocessBar')?.classList.contains('flex')", timeout=900000)
        page.wait_for_function("() => !generationUiActive", timeout=30000)
        preprocess_item = js_json(page, "ltxMotionPreprocessReady")
        preprocess_media = str(
            preprocess_item.get("mediaType")
            or preprocess_item.get("filename")
            or preprocess_item.get("path")
            or preprocess_item.get("url")
            or preprocess_item.get("image")
            or ""
        ).lower()
        if "video" not in preprocess_media and not preprocess_media.split("?", 1)[0].endswith((".mp4", ".webm", ".mov")):
            raise AssertionError(f"motion-{mode}: preprocess did not return a video preview: {preprocess_item!r}")
        preprocess_path = str(preprocess_item.get("path") or preprocess_item.get("filename") or "")
        if not preprocess_path.replace("\\", "/").startswith("Motion_Transfer/"):
            raise AssertionError(f"motion-{mode}: preprocess output was not stored under Motion_Transfer: {preprocess_item!r}")

    final_payload = collect_payload(page)
    assert_common_payload(final_payload, f"motion-{mode}-generate")
    final_video = final_payload.get("video")
    if not isinstance(final_video, dict) or final_video.get("motion_transfer_control_mode") != mode:
        raise AssertionError(f"motion-{mode}: final payload lost the selected mode: {final_payload!r}")
    distilled = final_payload.get("distilled_loras")
    if not isinstance(distilled, list) or len(distilled) != 1 or "384" not in str(distilled[0].get("name") if isinstance(distilled[0], dict) else ""):
        raise AssertionError(f"motion-{mode}: expected official IC-LoRA distilled 384 stack only, got {distilled!r}")
    if mode == "camera":
        job_id = click_global_generate(page)
    else:
        page.locator("#ltxMotionPreprocessBar button", has_text="Generate").click()
        page.wait_for_function("() => activeGenerationJobId", timeout=30000)
        job_id = str(page.evaluate("() => activeGenerationJobId"))
    job = poll_job(page, job_id)
    final_path = local_output_path(final_video_output(job, f"motion-{mode}"))
    quality = analyze_video_visual_quality(final_path, f"motion_{mode}")
    motion_quality = analyze_ltx_motion_transfer_motion(final_path, f"motion_{mode}", guide_path)
    page.wait_for_function("() => !generationUiActive", timeout=30000)
    gallery_item = latest_video_metadata(page)
    metadata = gallery_item.get("metadata") or gallery_item.get("info") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, dict):
        stored_video = metadata.get("video") or {}
        if isinstance(stored_video, dict) and stored_video.get("motion_transfer_control_mode") != mode:
            raise AssertionError(f"motion-{mode}: backend gallery metadata lost mode: {stored_video!r}")
    page.screenshot(path=str(RESULTS / f"app-smoke-ltx-motion-transfer-{mode}.png"), full_page=True)
    return {"mode": mode, "preprocess": preprocess_item, "job": job, "gallery": gallery_item, "quality": quality, "motion_quality": motion_quality}


def run_start_end(page: Page) -> dict[str, object]:
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          clearLtxMotionPreprocess({ quiet: true });
          syncLtxMotionTransferToggle(false);
        }"""
    )
    page.locator("#referenceImageInput").set_input_files([str(START_REFERENCE), str(END_REFERENCE)])
    force_battery_fields(page)
    page.evaluate(
        """() => {
          if (document.querySelector('#latentUpscaleSelect')) document.querySelector('#latentUpscaleSelect').value = 'Automatic';
          if (document.querySelector('#ltxLatentUpscaleRefineToggle')) document.querySelector('#ltxLatentUpscaleRefineToggle').checked = true;
          syncGenerationActionUi();
        }"""
    )
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.img2img?.reference_images?.length === 2
            && !payload?.img2img?.base_video
            && payload?.video?.transition_lora_enabled === true
            && payload?.video?.motion_transfer_enabled === false;
        }""",
        timeout=60000,
    )
    payload = collect_payload(page)
    assert_common_payload(payload, "start-end")
    video = payload.get("video")
    img2img = payload.get("img2img")
    if not isinstance(video, dict) or video.get("transition_lora_enabled") is not True:
        raise AssertionError(f"start/end: transition route was not enabled: {payload!r}")
    transition_lora = str(video.get("transition_lora") or "")
    if "transition" not in transition_lora.lower() and transition_lora.lower() not in {"automatic", "auto"}:
        raise AssertionError(f"start/end: transition LoRA is not selected/automatic: {video!r}")
    if not isinstance(img2img, dict) or len(img2img.get("reference_images") or []) != 2:
        raise AssertionError(f"start/end: expected start and end references: {payload!r}")
    job_id = click_global_generate(page)
    job = poll_job(page, job_id)
    final_path = local_output_path(final_video_output(job, "start-end"))
    quality = analyze_video_visual_quality(final_path, "start_end")
    continuity = analyze_start_end_continuity(final_path, payload)
    page.wait_for_function("() => !generationUiActive", timeout=30000)
    gallery_item = latest_video_metadata(page)
    metadata = gallery_item.get("metadata") or gallery_item.get("info") or {}
    if isinstance(metadata, str):
        try:
            metadata = json.loads(metadata)
        except json.JSONDecodeError:
            metadata = {}
    if isinstance(metadata, dict):
        stored_video = metadata.get("video") or {}
        if isinstance(stored_video, dict) and stored_video.get("transition_lora_enabled") is not True:
            raise AssertionError(f"start/end: backend gallery metadata lost transition route: {stored_video!r}")
    page.screenshot(path=str(RESULTS / "app-smoke-ltx-linear-start-end.png"), full_page=True)
    return {"payload": payload, "job": job, "gallery": gallery_item, "quality": quality, "continuity": continuity}


def assert_reference_workflow_sync(page: Page) -> None:
    workflows = backend_fetch(page, "/workflows")
    if not isinstance(workflows, list):
        raise AssertionError(f"Frontend could not sync workflows from backend: {workflows!r}")
    target = next((workflow for workflow in workflows if "ltx23_ic_lora" in str(workflow.get("id", "")).lower()), None)
    if not target:
        return
    analysis = backend_fetch(page, f"/workflows/{target['id']}/analysis")
    if not isinstance(analysis, dict) or not analysis.get("workflow"):
        raise AssertionError(f"Frontend could not read backend workflow analysis: {analysis!r}")
    classes = set(analysis.get("workflow", {}).get("class_types") or [])
    required = {"LTXVImgToVideoConditionOnly", "LTXAddVideoICLoRAGuide", "LTXICLoRALoaderModelOnly"}
    missing = required.difference(classes)
    if missing:
        raise AssertionError(f"Backend synced reference workflow is missing {sorted(missing)}.")


def main() -> None:
    require_inputs()
    summary: dict[str, object] = {
        "resolution": [WIDTH, HEIGHT],
        "steps": STEPS,
        "cfg": CFG,
        "detailer_enabled": ENABLE_DETAILER,
        "run_motion": RUN_MOTION,
        "run_start_end": RUN_START_END,
        "motion_guide": str(MOTION_GUIDE),
        "camera_motion_guide": str(CAMERA_MOTION_GUIDE),
        "target": str(TARGET_IMAGE),
        "start": str(START_REFERENCE),
        "end": str(END_REFERENCE),
        "cases": [],
    }
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.set_default_timeout(3600000)
        page.on("dialog", lambda dialog: dialog.accept())
        health = wait_front_backend_sync(page)
        summary["health"] = health
        set_ltx_linear_defaults(page)
        enable_ltx_detailer(page)
        assert_reference_workflow_sync(page)
        cases: list[dict[str, object]] = []
        if RUN_MOTION:
            for mode in MODES:
                cases.append(run_motion_transfer_mode(page, mode))
        if RUN_START_END:
            cases.append({"case": "start_end", **run_start_end(page)})
        summary["cases"] = cases
        browser.close()

    out = RESULTS / "front_ltx23_linear_motion_transfer_and_start_end_battery.json"
    out.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(f"ok frontend/backend LTX 2.3 linear battery: {out}")


if __name__ == "__main__":
    main()
