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
START = ROOT / "input" / "Smoke_splashART.jpeg"
END = ROOT / "input" / "Smoke_EndFrame.png"
WIDTH = 928
HEIGHT = 480
FPS = 24
SECONDS = 2
FRAMES = 49


def js_json(page: Page, expression: str):
    return json.loads(page.evaluate(f"async () => JSON.stringify(await ({expression}))"))


def backend_fetch(page: Page, path: str):
    return js_json(page, f"nexusFetch('{path}')")


def wait_front(page: Page) -> None:
    page.goto(BASE, wait_until="networkidle", timeout=60000)
    page.wait_for_function("() => typeof collectGenerationPayload === 'function' && backendOnline === true", timeout=120000)


def configure(page: Page, case: str) -> dict[str, object]:
    page.locator("button[data-preset='Wan']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'Wan' && currentActivity === 'img2img'", timeout=60000)
    page.locator("#tab-viewer").click()
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncWanLoopCycleToggle === 'function') syncWanLoopCycleToggle(false);
          document.querySelector('#posPrompt').value = 'cinematic smoke splash artwork, clean temporal motion, preserve composition';
          document.querySelector('#negPrompt').value = 'blur, deformation, noise, black frame';
          document.querySelector('#widthInput').value = '928';
          document.querySelector('#heightInput').value = '480';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#fpsInput').value = '24';
          document.querySelector('#secondsInput').value = '2';
          document.querySelector('#framesInput').value = '49';
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
    files = [str(START), str(END)] if case == "start_end" else str(START)
    page.locator("#referenceImageInput").set_input_files(files)
    expected_refs = 2 if case == "start_end" else 1
    page.wait_for_function(
        """expectedRefs => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'Wan'
            && payload?.activity === 'img2img'
            && payload?.img2img?.reference_images?.length === expectedRefs
            && payload?.video?.wan_loop_cycle === false
            && payload?.video?.fps === 24
            && payload?.video?.frames === 49
            && payload?.workflow_id == null
            && payload?.workflow_override == null;
        }""",
        arg=expected_refs,
        timeout=60000,
    )
    return js_json(page, "collectGenerationPayload()")


def click_generate(page: Page) -> str:
    with page.expect_request("**/api/generate/start"):
        page.locator("#globalGenerateButton").click()
        page.wait_for_function("() => activeGenerationJobId", timeout=30000)
    return str(page.evaluate("() => activeGenerationJobId"))


def poll(page: Page, job_id: str, timeout_ms: int = 900_000) -> dict[str, object]:
    deadline = time.monotonic() + timeout_ms / 1000
    last: dict[str, object] = {}
    while time.monotonic() < deadline:
        last = backend_fetch(page, f"/generate/{job_id}")
        if last.get("status") in {"completed", "failed", "cancelled"}:
            break
        time.sleep(3)
    else:
        raise TimeoutError(f"{job_id} timed out: {last!r}")
    if last.get("status") != "completed":
        raise AssertionError(f"{job_id} failed: {last!r}")
    return last


def local_path(output: dict[str, object]) -> Path:
    return ROOT / "output" / Path(str(output.get("path") or output.get("filename")).replace("\\", "/"))


def final_video(job: dict[str, object]) -> Path:
    videos = [
        item
        for item in (job.get("outputs") or [])
        if isinstance(item, dict) and str(item.get("path") or item.get("filename") or "").lower().endswith((".mp4", ".webm", ".mov"))
    ]
    if not videos:
        raise AssertionError(f"No video output: {job!r}")
    return local_path(videos[-1])


def make_strip(video_path: Path, case: str) -> Path:
    frame_dir = RESULTS / f"frames_wan22_{case}"
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
            "select='eq(n,0)+eq(n,12)+eq(n,24)+eq(n,36)+eq(n,48)',scale=186:96",
            "-vsync",
            "0",
            str(frame_dir / "frame_%02d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))
    strip = Image.new("RGB", (186 * len(frames), 122), (10, 10, 12))
    draw = ImageDraw.Draw(strip)
    for index, frame in enumerate(frames):
        img = Image.open(frame).convert("RGB")
        strip.paste(img, (186 * index, 0))
        draw.text((186 * index + 8, 102), f"f{index}", fill=(235, 235, 245))
    out = RESULTS / f"wan22-{case}-visual-strip.png"
    strip.save(out)
    return out


def run_case(page: Page, case: str) -> dict[str, object]:
    payload = configure(page, case)
    job_id = click_generate(page)
    job = poll(page, job_id)
    video = final_video(job)
    metrics = analyze_video(video, f"wan22_{case}", frames=8, require_motion=False)
    strip = make_strip(video, case)
    return {"case": case, "job_id": job_id, "output": str(video), "stream": ffprobe_video(video), "metrics": metrics, "strip": str(strip), "payload_refs": len((payload.get("img2img") or {}).get("reference_images") or [])}


def main() -> None:
    for path in (START, END):
        if not path.exists():
            raise AssertionError(f"Missing input: {path}")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1600, "height": 950})
        page = context.new_page()
        wait_front(page)
        for case in ("start", "start_end"):
            results.append(run_case(page, case))
        page.screenshot(path=str(RESULTS / "wan22-normal-routes-front.png"), full_page=True)
        browser.close()
    (RESULTS / "wan22-normal-routes-real.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
