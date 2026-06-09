import json
import os
import subprocess
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright

from smoke_ltx_motion_transfer_contract import (
    BASE,
    ROOT,
    RESULTS,
    backend_fetch,
    collect_payload,
    final_video_output,
    local_output_path,
)


START_IMAGE = Path(os.environ.get("NEXUS_LTX_START_IMAGE") or str(ROOT / "input" / "smoketestFirstFrame.png"))
END_IMAGE = Path(os.environ.get("NEXUS_LTX_END_IMAGE") or str(ROOT / "input" / "SmoketTestEndFrame.png"))
WIDTH = int(os.environ.get("NEXUS_LTX_WIDTH") or "512")
HEIGHT = int(os.environ.get("NEXUS_LTX_HEIGHT") or "512")
STEPS = [int(item.strip()) for item in os.environ.get("NEXUS_LTX_STEPS_SET", "8").split(",") if item.strip()]
CFG = float(os.environ.get("NEXUS_LTX_CFG") or "1")
FPS = float(os.environ.get("NEXUS_LTX_FPS") or "24")
SECONDS = float(os.environ.get("NEXUS_LTX_SECONDS") or "3")
MODEL_HINT = os.environ.get("NEXUS_LTX_MODEL_HINT", "").strip()
CASE_SUFFIX = os.environ.get("NEXUS_LTX_CASE_SUFFIX", "start_end").strip() or "start_end"
PROMPT = os.environ.get(
    "NEXUS_LTX_PROMPT",
    "same woman, clean cinematic transition from first frame to end frame, natural head turn, stable face, preserve hair and lighting",
)
NEGATIVE = os.environ.get("NEXUS_LTX_NEGATIVE", "noise, artifacts, speckled skin, smeared face, flicker, black frames")


def wait_front(page: Page) -> None:
    page.goto(BASE, wait_until="networkidle", timeout=60000)
    page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
    page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)
    health = backend_fetch(page, "/health")
    if health.get("nexus") != "ok":
        raise AssertionError(f"backend health failed through frontend: {health!r}")


def select_option_by_hint(page: Page, selector: str, hint: str) -> str:
    return page.evaluate(
        """([selector, hint]) => {
          const select = document.querySelector(selector);
          if (!select || !hint) return select?.value || '';
          const needle = String(hint || '').toLowerCase();
          const match = [...select.options].find(option => [option.value, option.textContent, option.dataset.model]
            .filter(Boolean)
            .some(value => String(value).toLowerCase().includes(needle)));
          if (match) {
            select.value = match.value;
            select.dispatchEvent(new Event('change', { bubbles: true }));
            return match.value;
          }
          return select.value || '';
        }""",
        [selector, hint],
    )


def configure(page: Page, steps: int) -> dict[str, object]:
    page.locator("button[data-preset='LTX']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'LTX' && currentActivity === 'img2img'", timeout=60000)
    if MODEL_HINT:
        select_option_by_hint(page, "#modelSelect", MODEL_HINT)
    page.locator("#tab-viewer").click()
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncLtxMotionTransferToggle === 'function') syncLtxMotionTransferToggle(false);
          if (typeof syncLtxLoopCycleToggle === 'function') syncLtxLoopCycleToggle(false);
          activeWorkflowId = null;
          activeWorkflowGraph = null;
          activeWorkflowAnalysis = null;
        }"""
    )
    for selector, value in [
        ("#widthInput", WIDTH),
        ("#heightInput", HEIGHT),
        ("#stepsValue", steps),
        ("#cfgValue", CFG),
        ("#secondsInput", SECONDS),
        ("#fpsInput", FPS),
        ("#posPrompt", PROMPT),
        ("#negPrompt", NEGATIVE),
    ]:
        page.locator(selector).fill(str(value))
    page.evaluate(
        """() => {
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          const denoise = document.querySelector('#denoiseSlider');
          if (denoise) denoise.value = '0.85';
          updateSliderFromNumber('denoise');
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler CFG++';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Quadratic';
          const latent = document.querySelector('#latentUpscaleSelect');
          if (latent) latent.value = 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors';
          const refine = document.querySelector('#ltxLatentUpscaleRefineToggle');
          if (refine) refine.checked = true;
          const d1 = document.querySelector('#distilledLoraOneSelect');
          const d2 = document.querySelector('#distilledLoraTwoSelect');
          if (d1) d1.value = 'ltx\\\\ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors';
          if (d2) d2.value = 'ltx\\\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors';
          if (document.querySelector('#distilledLoraOneStrength')) document.querySelector('#distilledLoraOneStrength').value = '0.8';
          if (document.querySelector('#distilledLoraTwoStrength')) document.querySelector('#distilledLoraTwoStrength').value = '0.5';
          const detailer = document.querySelector('#ltxDetailerToggle');
          if (detailer) detailer.checked = false;
          const detailerLora = document.querySelector('#ltxDetailerLoraSelect');
          if (detailerLora) detailerLora.value = 'None';
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
    )
    page.locator("#referenceImageInput").set_input_files([str(START_IMAGE), str(END_IMAGE)])
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'LTX'
            && payload?.activity === 'img2img'
            && payload?.workspace === 'viewer'
            && payload?.img2img?.reference_images?.length === 2
            && payload?.video?.transition_lora_enabled === true
            && payload?.video?.motion_transfer_enabled === false
            && payload?.video?.ltx_loop_cycle === false
            && payload?.width === __WIDTH__
            && payload?.height === __HEIGHT__;
        }""".replace("__WIDTH__", str(WIDTH)).replace("__HEIGHT__", str(HEIGHT)),
        timeout=60000,
    )
    return collect_payload(page)


def extract_frames(video_path: Path, case: str) -> Path:
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
            str(video_path),
            "-vf",
            "select='eq(n,0)+eq(n,18)+eq(n,36)+eq(n,54)+eq(n,72)'",
            "-vsync",
            "0",
            str(frame_dir / "frame_%02d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise AssertionError(f"{case}: no full-size frames extracted")
    first = Image.open(frames[0]).convert("RGB")
    canvas = Image.new("RGB", (first.width * len(frames), first.height + 28), (10, 10, 12))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        img = Image.open(frame).convert("RGB")
        canvas.paste(img, (index * first.width, 0))
        draw.text((index * first.width + 6, first.height + 7), frame.name, fill=(235, 235, 235))
    strip = RESULTS / f"{case}_fullsize_strip.png"
    canvas.save(strip)
    return strip


def run_case(page: Page, steps: int) -> dict[str, object]:
    case = f"ltx23_{CASE_SUFFIX}_{steps}steps"
    payload = configure(page, steps)
    job = page.evaluate(
        """async () => {
          const payload = collectGenerationPayload();
          const job = await startGenerationJob(payload);
          return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
        }"""
    )
    output = final_video_output(job, case)
    video_path = local_output_path(output)
    strip = extract_frames(video_path, case)
    return {"case": case, "payload": payload, "output": str(video_path), "strip": str(strip)}


def main() -> None:
    if not START_IMAGE.exists() or not END_IMAGE.exists():
        raise AssertionError(f"missing start/end input: {START_IMAGE} / {END_IMAGE}")
    started = time.time()
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        wait_front(page)
        for steps in STEPS:
            results.append(run_case(page, steps))
        browser.close()
    out = RESULTS / f"ltx23_start_end_{CASE_SUFFIX}_visual.json"
    out.write_text(json.dumps({"started": started, "results": results}, indent=2), encoding="utf-8")
    print("ok ltx23 start/end visual: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
