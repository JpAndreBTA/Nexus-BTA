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
from visual_checks import analyze_video


INPUT_IMAGE = Path(os.environ.get("NEXUS_LTX_INPUT_IMAGE") or str(ROOT / "input" / "SmokeTest.png"))
SOURCE_WIDTH, SOURCE_HEIGHT = Image.open(INPUT_IMAGE).size if INPUT_IMAGE.exists() else (912, 512)
WIDTH = int(os.environ.get("NEXUS_LTX_WIDTH") or (SOURCE_WIDTH - SOURCE_WIDTH % 32))
HEIGHT = int(os.environ.get("NEXUS_LTX_HEIGHT") or (SOURCE_HEIGHT - SOURCE_HEIGHT % 32))
STEPS = [int(item.strip()) for item in os.environ.get("NEXUS_LTX_STEPS_SET", "4,8").split(",") if item.strip()]
CFG = float(os.environ.get("NEXUS_LTX_CFG") or "1")
FPS = float(os.environ.get("NEXUS_LTX_FPS") or "24")
SECONDS = float(os.environ.get("NEXUS_LTX_SECONDS") or "2")
DENOISE = float(os.environ.get("NEXUS_LTX_DENOISE") or "1")
PROMPT = os.environ.get(
    "NEXUS_LTX_PROMPT",
    "preserve the same seated chef portrait, blue outfit, chef hat, warm curtains, chair and room, natural subtle motion",
)
NEGATIVE = os.environ.get("NEXUS_LTX_NEGATIVE", "noise, artifacts, smeared face, deformed hands, unrecognizable subject")
LATENT_UPSCALE = os.environ.get("NEXUS_LTX_LATENT_UPSCALE", "ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
REQUIRE_MOTION = os.environ.get("NEXUS_LTX_REQUIRE_MOTION", "0").lower() in {"1", "true", "yes", "y"}
CASE_SUFFIX = os.environ.get("NEXUS_LTX_CASE_SUFFIX", "smoketest").strip() or "smoketest"
MODEL_HINT = os.environ.get("NEXUS_LTX_MODEL_HINT", "").strip()
DISTILLED_ONE = os.environ.get("NEXUS_LTX_DISTILLED_ONE", r"ltx\ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors")
DISTILLED_TWO = os.environ.get("NEXUS_LTX_DISTILLED_TWO", r"ltx\ltx-2.3-22b-distilled-lora-384-1.1.safetensors")
DISTILLED_ONE_STRENGTH = os.environ.get("NEXUS_LTX_DISTILLED_ONE_STRENGTH", "0.8")
DISTILLED_TWO_STRENGTH = os.environ.get("NEXUS_LTX_DISTILLED_TWO_STRENGTH", "0.5")
DETAILER = os.environ.get("NEXUS_LTX_DETAILER", "off").strip().lower() in {"1", "true", "yes", "on", "detailer"}
OMNICINE = os.environ.get("NEXUS_LTX_OMNICINE", "off").strip().lower() in {"1", "true", "yes", "on", "omnicine"}
STRICT_METRICS = os.environ.get("NEXUS_LTX_STRICT_METRICS", "0").strip().lower() in {"1", "true", "yes", "on", "strict"}


def wait_front(page: Page) -> None:
    page.goto(BASE, wait_until="networkidle", timeout=60000)
    page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
    page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)
    health = backend_fetch(page, "/health")
    if health.get("nexus") != "ok":
        raise AssertionError(f"backend health failed through frontend: {health!r}")


def set_value(page: Page, selector: str, value: object) -> None:
    page.locator(selector).fill(str(value))


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


def configure_ltx(page: Page, steps: int) -> dict[str, object]:
    page.locator("button[data-preset='LTX']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'LTX' && currentActivity === 'img2img'", timeout=60000)
    if MODEL_HINT:
        select_option_by_hint(page, "#modelSelect", MODEL_HINT)
    page.locator("#tab-viewer").click()
    page.wait_for_function("() => activeWorkspace === 'viewer'", timeout=30000)
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncLtxMotionTransferToggle === 'function') syncLtxMotionTransferToggle(false);
          const mode = document.querySelector('#img2imgModeSelect');
          if (mode) mode.value = 'Image to Image';
          activeWorkflowId = null;
          activeWorkflowGraph = null;
          activeWorkflowAnalysis = null;
        }"""
    )
    set_value(page, "#widthInput", WIDTH)
    set_value(page, "#heightInput", HEIGHT)
    set_value(page, "#stepsValue", steps)
    set_value(page, "#cfgValue", CFG)
    set_value(page, "#secondsInput", SECONDS)
    set_value(page, "#fpsInput", FPS)
    set_value(page, "#posPrompt", PROMPT)
    set_value(page, "#negPrompt", NEGATIVE)
    page.evaluate(
        """([denoiseValue, latentUpscale, distilledOne, distilledTwo, strengthOne, strengthTwo, detailerEnabled, omnicineEnabled]) => {
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          const denoise = document.querySelector('#denoiseSlider');
          if (denoise) denoise.value = String(denoiseValue);
          updateSliderFromNumber('denoise');
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler CFG++';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Quadratic';
          const d1 = document.querySelector('#distilledLoraOneSelect');
          const d2 = document.querySelector('#distilledLoraTwoSelect');
          if (d1) d1.value = distilledOne;
          if (d2) d2.value = distilledTwo;
          const s1 = document.querySelector('#distilledLoraOneStrength');
          const s2 = document.querySelector('#distilledLoraTwoStrength');
          if (s1) s1.value = strengthOne;
          if (s2) s2.value = strengthTwo;
          const latent = document.querySelector('#latentUpscaleSelect');
          if (latent) latent.value = String(latentUpscale);
          const refine = document.querySelector('#ltxLatentUpscaleRefineToggle');
          if (refine) refine.checked = true;
          const detailer = document.querySelector('#ltxDetailerToggle');
          if (detailer) detailer.checked = !!detailerEnabled;
          const detailerLora = document.querySelector('#ltxDetailerLoraSelect');
          if (detailerLora) detailerLora.value = detailerEnabled ? 'Automatic' : 'None';
          const omnicine = document.querySelector('#omnicineSelect');
          if (omnicine) omnicine.value = omnicineEnabled ? 'Automatic' : 'Off';
          if (typeof updateOmnicineStatus === 'function') updateOmnicineStatus();
          syncGenerationActionUi();
          updateWorkflowPreview();
        }""",
        [
            DENOISE,
            LATENT_UPSCALE,
            DISTILLED_ONE,
            DISTILLED_TWO,
            DISTILLED_ONE_STRENGTH,
            DISTILLED_TWO_STRENGTH,
            DETAILER,
            OMNICINE,
        ],
    )
    page.locator("#referenceImageInput").set_input_files(str(INPUT_IMAGE))
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'LTX'
            && payload?.activity === 'img2img'
            && payload?.workspace === 'viewer'
            && payload?.img2img?.reference_images?.length === 1
            && !payload?.img2img?.base_video
            && payload?.workflow_id == null
            && payload?.video?.motion_transfer_enabled === false;
        }""",
        timeout=60000,
    )
    payload = collect_payload(page)
    if payload.get("width") != WIDTH or payload.get("height") != HEIGHT:
        raise AssertionError(f"dimension sync failed: expected {WIDTH}x{HEIGHT}, got {payload.get('width')}x{payload.get('height')}")
    if payload.get("steps") != steps or float(payload.get("cfg")) != CFG:
        raise AssertionError(f"step/cfg sync failed: {payload!r}")
    return payload


def make_strip(video_path: Path, case: str) -> Path:
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
            "select='eq(n,0)+eq(n,12)+eq(n,24)+eq(n,36)+eq(n,48)'",
            "-vsync",
            "0",
            str(frame_dir / "frame_%02d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise AssertionError(f"{case}: no strip frames extracted")
    first = Image.open(frames[0]).convert("RGB")
    tile_w, tile_h = first.size
    canvas = Image.new("RGB", (tile_w * len(frames), tile_h + 28), (10, 10, 12))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        img = Image.open(frame).convert("RGB")
        canvas.paste(img, (index * tile_w, 0))
        draw.text((index * tile_w + 6, tile_h + 7), f"f{index}", fill=(235, 235, 235))
    strip = RESULTS / f"{case}_strip.png"
    canvas.save(strip)
    return strip


def run_case(page: Page, steps: int) -> dict[str, object]:
    case = f"ltx23_linear_{CASE_SUFFIX}_{steps}steps"
    payload = configure_ltx(page, steps)
    job = page.evaluate(
        """async () => {
          const payload = collectGenerationPayload();
          const job = await startGenerationJob(payload);
          return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
        }"""
    )
    output = final_video_output(job, case)
    video_path = local_output_path(output)
    strip = make_strip(video_path, case)
    metric_warning = ""
    try:
        metrics = analyze_video(video_path, case, frames=8, require_motion=REQUIRE_MOTION)
    except AssertionError as exc:
        if STRICT_METRICS:
            raise
        metric_warning = str(exc)
        metrics = {"warning": metric_warning}
    return {"case": case, "payload": payload, "output": str(video_path), "strip": str(strip), "metrics": metrics}


def main() -> None:
    if not INPUT_IMAGE.exists():
        raise AssertionError(f"missing input: {INPUT_IMAGE}")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        wait_front(page)
        for steps in STEPS:
            results.append(run_case(page, steps))
        browser.close()
    (RESULTS / "ltx23_linear_smoketest_visual.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ok ltx23 smoketest visual: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
