import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright

from visual_checks import analyze_video, ffprobe_video


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
SAMPLE = ROOT / "input" / "Smoke_splashART.jpeg"
WIDTH = 928
HEIGHT = 480
FPS = 24
SECONDS = 5
FRAMES = 121
STEPS = 4
CFG = 1


def js_json(page: Page, expression: str):
    return json.loads(page.evaluate(f"async () => JSON.stringify(await ({expression}))"))


def backend_fetch(page: Page, path: str):
    return js_json(page, f"nexusFetch('{path}')")


def wait_front_backend_sync(page: Page) -> None:
    page.goto(BASE, wait_until="networkidle", timeout=60000)
    page.wait_for_function("() => typeof collectGenerationPayload === 'function' && typeof nexusFetch === 'function'", timeout=120000)
    page.wait_for_function("() => backendOnline === true", timeout=120000)
    health = backend_fetch(page, "/health")
    if health.get("nexus") != "ok":
        raise AssertionError(f"Backend health failed: {health!r}")


def collect_payload(page: Page) -> dict[str, object]:
    return js_json(page, "collectGenerationPayload()")


def configure_ltx_loop(page: Page) -> dict[str, object]:
    latent_mode = os.environ.get("NEXUS_LTX_LOOP_LATENT", "on").strip().lower()
    detailer_mode = os.environ.get("NEXUS_LTX_LOOP_DETAILER", "on").strip().lower()
    omnicine_mode = os.environ.get("NEXUS_LTX_LOOP_OMNICINE", "off").strip().lower()
    livewallpaper_mode = os.environ.get("NEXUS_LTX_LOOP_LIVEWALLPAPER", "off").strip().lower()
    audio_mode = os.environ.get("NEXUS_LTX_LOOP_AUDIO", "off").strip().lower()
    latent_enabled = latent_mode in {"1", "true", "yes", "on", "x2", "upscale"}
    detailer_enabled = detailer_mode in {"1", "true", "yes", "on", "detailer"}
    omnicine_enabled = omnicine_mode in {"1", "true", "yes", "on", "omnicine"}
    audio_enabled = audio_mode in {"1", "true", "yes", "on", "audio"}
    page.locator("button[data-preset='LTX']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'LTX' && currentActivity === 'img2img'", timeout=60000)
    page.locator("#tab-viewer").click()
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncLtxMotionTransferToggle === 'function') syncLtxMotionTransferToggle(false);
          if (typeof syncLtxLoopCycleToggle === 'function') syncLtxLoopCycleToggle(true);
          document.querySelector('#posPrompt').value = 'cinematic splash art forward seamless loop, hair and ribbons sway, neon smoke curls, energy blade pulses, tiny sparks drift, subtle parallax, visible continuous cyclical motion, preserve composition';
          document.querySelector('#negPrompt').value = 'noise, blur, jump cut, flicker, black frame, identity drift, different person, frozen frame, static hold, pingpong loop, boomerang, reverse motion';
          document.querySelector('#widthInput').value = '928';
          document.querySelector('#heightInput').value = '480';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#fpsInput').value = '24';
          document.querySelector('#secondsInput').value = '5';
          document.querySelector('#framesInput').value = '121';
          const model = document.querySelector('#modelSelect');
          const daiswa = model ? [...model.options].find(option => /DasiwaLTX23Lightspeed_solsticecoinV2/i.test(option.value) || /DasiwaLTX23Lightspeed_solsticecoinV2/i.test(option.textContent || '')) : null;
          if (!daiswa) throw new Error('Missing Dasiwa LTX checkpoint in modelSelect');
          model.value = daiswa.value;
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler CFG++';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Quadratic';
          const latent = document.querySelector('#latentUpscaleSelect');
          const latentEnabled = "__LATENT_ENABLED__" === 'true';
          if (latent) latent.value = latentEnabled ? 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors' : 'None';
          const refine = document.querySelector('#ltxLatentUpscaleRefineToggle');
          if (refine) refine.checked = latentEnabled;
          const detailer = document.querySelector('#ltxDetailerToggle');
          const detailerEnabled = "__DETAILER_ENABLED__" === 'true';
          if (detailer) detailer.checked = detailerEnabled;
          const detailerLora = document.querySelector('#ltxDetailerLoraSelect');
          if (detailerLora) detailerLora.value = detailerEnabled ? 'Automatic' : 'None';
          const detailerStrength = document.querySelector('#ltxDetailerStrengthSlider');
          if (detailerStrength) detailerStrength.value = '0.85';
          const activeAudio = document.querySelector('#activeAudioToggle');
          const audioEnabled = "__AUDIO_ENABLED__" === 'true';
          if (activeAudio) activeAudio.checked = audioEnabled;
          const audioVae = document.querySelector('#audioVaeSelect');
          if (audioVae && audioEnabled) {
            const option = [...audioVae.options].find(item => /LTX23_audio_vae_bf16/i.test(`${item.value} ${item.textContent || ''}`));
            if (option) audioVae.value = option.value;
          }
          const omni = document.querySelector('#omnicineSelect');
          const omniEnabled = "__OMNICINE_ENABLED__" === 'true';
          if (omni) {
            if (omniEnabled) {
              const option = [...omni.options].find(item => /omnicine|singularity/i.test(`${item.value} ${item.textContent || ''}`));
              if (!option) throw new Error('Missing OmniCine LoRA option in omnicineSelect');
              omni.value = option.value;
            } else {
              omni.value = 'Off';
            }
            if (typeof updateOmnicineStatus === 'function') updateOmnicineStatus();
          } else if (omniEnabled) {
            throw new Error('Missing omnicineSelect control');
          }
          const d1 = document.querySelector('#distilledLoraOneSelect');
          const d2 = document.querySelector('#distilledLoraTwoSelect');
          if (d1) d1.value = 'None';
          if (d2) d2.value = 'ltx\\\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors';
          if (document.querySelector('#distilledLoraTwoStrength')) document.querySelector('#distilledLoraTwoStrength').value = '0.5';
          if ("__LIVEWALLPAPER_MODE__" === 'on') {
            activeLoras = [{ name: 'ltx\\\\livewallpaper_ltx23_r64_6250.safetensors', relative_name: 'ltx\\\\livewallpaper_ltx23_r64_6250.safetensors', strength: 0.35, strength_model: 0.35, strength_clip: 0 }];
            renderActiveLoras();
          }
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          syncVideoMotionFields('framesInput');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
        .replace("__LATENT_ENABLED__", "true" if latent_enabled else "false")
        .replace("__DETAILER_ENABLED__", "true" if detailer_enabled else "false")
        .replace("__OMNICINE_ENABLED__", "true" if omnicine_enabled else "false")
        .replace("__AUDIO_ENABLED__", "true" if audio_enabled else "false")
        .replace("__LIVEWALLPAPER_MODE__", "on" if livewallpaper_mode in {"1", "true", "yes", "on"} else "off")
    )
    page.locator("#referenceImageInput").set_input_files(str(SAMPLE))
    latent_condition = "String(payload?.video?.latent_upscale || '').toLowerCase() !== 'none'" if latent_enabled else "['none','off','disabled','false','0','no'].includes(String(payload?.video?.latent_upscale || '').toLowerCase())"
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'LTX'
            && payload?.activity === 'img2img'
            && payload?.workspace === 'viewer'
            && payload?.img2img?.reference_images?.length === 2
            && payload?.width === 928
            && payload?.height === 480
            && /DasiwaLTX23Lightspeed_solsticecoinV2/i.test(`${payload?.model_name || ''} ${payload?.model_path || ''}`)
            && payload?.steps === 4
            && payload?.cfg === 1
            && payload?.video?.fps === 24
            && payload?.video?.seconds === 5
            && payload?.video?.frames === 121
            && payload?.video?.ltx_loop_cycle === true
            && payload?.video?.ltx_loop_source === 'start_frame_as_end_frame'
            && payload?.video?.transition_lora_enabled === false
            && payload?.video?.motion_strength === 0.30
            && payload?.video?.start_frame_strength === 0.70
            && payload?.video?.end_frame_strength === 0.70
            && __LATENT_CONDITION__
            && payload?.video?.latent_upscale_refine === __LATENT_ENABLED__
            && payload?.video?.detailer_enabled === __DETAILER_ENABLED__
            && payload?.video?.omnicine_enabled === __OMNICINE_ENABLED__
            && payload?.video?.active_audio === __AUDIO_ENABLED__
            && payload?.video?.ltx_endpoint_frame_lock === false
            && payload?.video?.ltx_loop_post_seam_blend === false
            && payload?.video?.ltx_loop_mid_motion_guide === false
            && payload?.workflow_id == null
            && payload?.workflow_override == null;
        }"""
        .replace("__LATENT_CONDITION__", latent_condition)
        .replace("__LATENT_ENABLED__", str(latent_enabled).lower())
        .replace("__DETAILER_ENABLED__", str(detailer_enabled).lower())
        .replace("__OMNICINE_ENABLED__", str(omnicine_enabled).lower())
        .replace("__AUDIO_ENABLED__", str(audio_enabled).lower()),
        timeout=60000,
    )
    return collect_payload(page)


def click_generate(page: Page) -> str:
    with page.expect_request("**/api/generate/start"):
        page.locator("#globalGenerateButton").click()
        page.wait_for_function("() => activeGenerationJobId", timeout=30000)
    return str(page.evaluate("() => activeGenerationJobId"))


def poll_job(page: Page, job_id: str, timeout_ms: int = 1_200_000) -> dict[str, object]:
    deadline = time.monotonic() + timeout_ms / 1000
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        last = backend_fetch(page, f"/generate/{job_id}")
        if last.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(3)
    else:
        raise TimeoutError(f"Job {job_id} timed out: {last!r}")
    if last.get("status") != "completed":
        raise AssertionError(f"Job {job_id} did not complete: {last!r}")
    return last


def local_output_path(output: dict[str, object]) -> Path:
    path = str(output.get("path") or output.get("filename") or "")
    if not path:
        raise AssertionError(f"Output has no path: {output!r}")
    return ROOT / "output" / Path(path.replace("\\", "/"))


def final_video_output(job: dict[str, object], started_at: float | None = None) -> dict[str, object]:
    outputs = job.get("outputs") or []
    videos = [
        item
        for item in outputs
        if isinstance(item, dict) and str(item.get("path") or item.get("filename") or "").lower().endswith((".mp4", ".webm", ".mov"))
    ]
    if videos:
        if any(_bool_payload_value((item.get("metadata") or {}).get("video", {}).get("active_audio")) for item in videos if isinstance(item.get("metadata"), dict)):
            videos = sorted(
                videos,
                key=lambda item: "-audio" in str(item.get("path") or item.get("filename") or "").lower(),
            )
        candidate = videos[-1]
        path = local_output_path(candidate)
        if started_at is None or (path.exists() and path.stat().st_mtime >= started_at - 5):
            return candidate
    if started_at is not None:
        recent = [
            path
            for path in (ROOT / "output" / "video").glob("*LTX*i2v*NEXUS_BTA_LTX23_IMG2VID_928x480*.mp4")
            if path.stat().st_mtime >= started_at - 5
        ]
        if recent:
            audio_recent = [path for path in recent if "-audio" in path.stem.lower()]
            path = max(audio_recent or recent, key=lambda item: item.stat().st_mtime)
            return {"path": str(path.relative_to(ROOT / "output")).replace("\\", "/")}
    raise AssertionError(f"No fresh video output in job: {job!r}")


def _bool_payload_value(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"", "false", "0", "off", "none", "no"}
    return bool(value)


def ffprobe_streams(path: Path) -> dict[str, object]:
    return json.loads(
        subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
            cwd=ROOT,
            text=True,
            encoding="utf-8",
        )
    )


def extract_frame(video_path: Path, selector: str, target: Path) -> Image.Image:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vf", selector, "-frames:v", "1", str(target)],
        cwd=ROOT,
        check=True,
    )
    return Image.open(target).convert("RGB")


def loop_metrics(video_path: Path, case: str) -> dict[str, object]:
    stream = ffprobe_video(video_path)
    try:
        frame_count = int(float(stream.get("nb_frames") or FRAMES))
    except (TypeError, ValueError):
        frame_count = FRAMES
    middle_index = max(1, min(frame_count - 2, frame_count // 2))
    last_index = max(1, frame_count - 1)
    frame_dir = RESULTS / f"frames_{case}_loop"
    first = extract_frame(video_path, "select='eq(n,0)',scale=192:192", frame_dir / "first.png")
    middle = extract_frame(video_path, f"select='eq(n,{middle_index})',scale=192:192", frame_dir / "middle.png")
    last = extract_frame(video_path, f"select='eq(n,{last_index})',scale=192:192", frame_dir / "last.png")
    arr_first = np.asarray(first, dtype=np.float32)
    arr_middle = np.asarray(middle, dtype=np.float32)
    arr_last = np.asarray(last, dtype=np.float32)
    seam_mad = float(np.mean(np.abs(arr_first - arr_last)))
    motion_mad = float(np.mean(np.abs(arr_first - arr_middle)))
    middle_std = float(arr_middle.std())
    diff = np.asarray(np.abs(arr_first - arr_last).clip(0, 255), dtype=np.uint8)
    strip = Image.new("RGB", (192 * 4, 220), (10, 10, 12))
    draw = ImageDraw.Draw(strip)
    for index, (title, image) in enumerate([("first", first), ("middle", middle), ("last", last), ("seam diff", Image.fromarray(diff).convert("RGB"))]):
        strip.paste(image, (192 * index, 0))
        draw.text((192 * index + 8, 198), title, fill=(235, 235, 245))
    strip_path = RESULTS / f"ltx23-loop-cycle-visual-strip-{case}.png"
    strip.save(strip_path)
    if seam_mad > max(42.0, motion_mad * 1.45):
        raise AssertionError(f"LTX loop seam too abrupt: seam={seam_mad:.2f} motion={motion_mad:.2f} strip={strip_path}")
    if motion_mad < 8.0:
        raise AssertionError(f"LTX loop has too little visible mid-clip motion: motion={motion_mad:.2f} strip={strip_path}")
    if middle_std < 45.0:
        raise AssertionError(f"LTX loop mid-frame looks washed/noisy instead of crisp splash art: std={middle_std:.2f} strip={strip_path}")
    return {"seam_mad": seam_mad, "motion_mad": motion_mad, "middle_std": middle_std, "middle_index": middle_index, "last_index": last_index, "strip": str(strip_path)}


def main() -> None:
    if not SAMPLE.exists():
        raise AssertionError(f"Missing smoke input: {SAMPLE}")
    case = os.environ.get("NEXUS_LTX_LOOP_CASE", "ltx23_loop_cycle").strip() or "ltx23_loop_cycle"
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 950})
        page = context.new_page()
        page.set_default_timeout(1_200_000)
        page.on("dialog", lambda dialog: dialog.accept())
        wait_front_backend_sync(page)
        payload = configure_ltx_loop(page)
        started_at = time.time()
        job_id = click_generate(page)
        job = poll_job(page, job_id)
        output = final_video_output(job, started_at)
        path = local_output_path(output)
        full_probe = ffprobe_streams(path)
        audio_streams = [stream for stream in full_probe.get("streams", []) if stream.get("codec_type") == "audio"]
        if _bool_payload_value(payload.get("video", {}).get("active_audio")) and not audio_streams:
            sibling = path.with_name(path.stem + "-audio" + path.suffix)
            if sibling.exists():
                path = sibling
                full_probe = ffprobe_streams(path)
                audio_streams = [stream for stream in full_probe.get("streams", []) if stream.get("codec_type") == "audio"]
        if _bool_payload_value(payload.get("video", {}).get("active_audio")) and not audio_streams:
            raise AssertionError(f"LTX loop active_audio generated no audio stream: {path}")
        metrics = analyze_video(path, case, frames=11, require_motion=True)
        loop = loop_metrics(path, case)
        page.screenshot(path=str(RESULTS / f"ltx23-loop-cycle-front-{case}.png"), full_page=True)
        browser.close()

    stream = ffprobe_video(path)
    result = {
        "payload": payload,
        "job_id": job_id,
        "output": str(path),
        "stream": stream,
        "streams": full_probe.get("streams", []),
        "metrics": metrics,
        "loop": loop,
    }
    (RESULTS / f"ltx23-loop-cycle-real-{case}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (RESULTS / "ltx23-loop-cycle-real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
