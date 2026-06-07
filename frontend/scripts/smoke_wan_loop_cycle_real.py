import json
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


def configure_wan_loop(page: Page) -> dict[str, object]:
    page.locator("button[data-preset='Wan']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'Wan' && currentActivity === 'img2img'", timeout=60000)
    page.locator("#tab-viewer").click()
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncWanLoopCycleToggle === 'function') syncWanLoopCycleToggle(true);
          document.querySelector('#posPrompt').value = 'cinematic smoke splash artwork, subtle cyclic motion, seamless loop, preserve composition';
          document.querySelector('#negPrompt').value = 'blur, deformation, jump cut, flicker, black frame';
          document.querySelector('#widthInput').value = '928';
          document.querySelector('#heightInput').value = '480';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#fpsInput').value = '24';
          document.querySelector('#secondsInput').value = '5';
          document.querySelector('#framesInput').value = '121';
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Simple';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          syncVideoMotionFields('framesInput');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
    )
    page.locator("#referenceImageInput").set_input_files(str(SAMPLE))
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'Wan'
            && payload?.activity === 'img2img'
            && payload?.img2img?.reference_images?.length === 1
            && payload?.width === 928
            && payload?.height === 480
            && payload?.steps === 4
            && payload?.cfg === 1
            && payload?.video?.fps === 24
            && payload?.video?.seconds === 5
            && payload?.video?.frames === 121
            && payload?.video?.wan_loop_cycle === true
            && payload?.video?.wan_loop_source === 'start_frame_as_end_frame';
        }""",
        timeout=60000,
    )
    return collect_payload(page)


def click_generate(page: Page) -> str:
    with page.expect_request("**/api/generate/start"):
        page.locator("#globalGenerateButton").click()
        page.wait_for_function("() => activeGenerationJobId", timeout=30000)
    return str(page.evaluate("() => activeGenerationJobId"))


def poll_job(page: Page, job_id: str, timeout_ms: int = 1_500_000) -> dict[str, object]:
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


def final_video_output(job: dict[str, object]) -> dict[str, object]:
    outputs = job.get("outputs") or []
    videos = [
        item
        for item in outputs
        if isinstance(item, dict) and str(item.get("path") or item.get("filename") or "").lower().endswith((".mp4", ".webm", ".mov"))
    ]
    if not videos:
        raise AssertionError(f"No video output in job: {job!r}")
    return videos[-1]


def extract_frame(video_path: Path, selector: str, target: Path) -> Image.Image:
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(video_path), "-vf", selector, "-frames:v", "1", str(target)],
        cwd=ROOT,
        check=True,
    )
    return Image.open(target).convert("RGB")


def loop_metrics(video_path: Path, case: str) -> dict[str, object]:
    frame_dir = RESULTS / f"frames_{case}_loop"
    first = extract_frame(video_path, "select='eq(n,0)',scale=232:120", frame_dir / "first.png")
    middle = extract_frame(video_path, "select='eq(n,60)',scale=232:120", frame_dir / "middle.png")
    last = extract_frame(video_path, "select='eq(n,120)',scale=232:120", frame_dir / "last.png")
    arr_first = np.asarray(first, dtype=np.float32)
    arr_middle = np.asarray(middle, dtype=np.float32)
    arr_last = np.asarray(last, dtype=np.float32)
    seam_mad = float(np.mean(np.abs(arr_first - arr_last)))
    motion_mad = float(np.mean(np.abs(arr_first - arr_middle)))
    diff = np.asarray(np.abs(arr_first - arr_last).clip(0, 255), dtype=np.uint8)
    diff_img = Image.fromarray(diff).convert("RGB")
    strip = Image.new("RGB", (232 * 4, 148), (10, 10, 12))
    draw = ImageDraw.Draw(strip)
    for index, (title, image) in enumerate([("first", first), ("middle", middle), ("last", last), ("seam diff", diff_img)]):
        strip.paste(image, (232 * index, 0))
        draw.text((232 * index + 8, 126), title, fill=(235, 235, 245))
    strip_path = RESULTS / "wan22-loop-cycle-visual-strip.png"
    strip.save(strip_path)
    if seam_mad > max(45.0, motion_mad * 1.35):
        raise AssertionError(f"Loop seam is visually too abrupt: seam={seam_mad:.2f} motion={motion_mad:.2f} strip={strip_path}")
    return {"seam_mad": seam_mad, "motion_mad": motion_mad, "strip": str(strip_path)}


def main() -> None:
    if not SAMPLE.exists():
        raise AssertionError(f"Missing smoke input: {SAMPLE}")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 950})
        page = context.new_page()
        wait_front_backend_sync(page)
        payload = configure_wan_loop(page)
        job_id = click_generate(page)
        job = poll_job(page, job_id)
        output = final_video_output(job)
        path = local_output_path(output)
        metrics = analyze_video(path, "wan22_loop_cycle", frames=10, require_motion=True)
        loop = loop_metrics(path, "wan22_loop_cycle")
        page.screenshot(path=str(RESULTS / "wan22-loop-cycle-front.png"), full_page=True)
        browser.close()

    stream = ffprobe_video(path)
    result = {"payload": payload, "job_id": job_id, "output": str(path), "stream": stream, "metrics": metrics, "loop": loop}
    (RESULTS / "wan22-loop-cycle-real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
