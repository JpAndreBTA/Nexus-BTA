import json
import os
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

from visual_checks import RESULTS, analyze_image, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
BASE_IMAGE = Path(os.environ.get("QWEN_SMOKE_BASE_IMAGE") or (ROOT / "input" / "SmokeTeste_BaseMultipleReference.png"))
TEST_DENOISE = os.environ.get("QWEN_SMOKE_DENOISE", "0.78")
TEST_CASES = {item.strip() for item in os.environ.get("QWEN_SMOKE_CASES", "linear,multiview,inpaint").split(",") if item.strip()}
TEST_WIDTH = int(os.environ.get("QWEN_SMOKE_WIDTH", "208"))
TEST_HEIGHT = int(os.environ.get("QWEN_SMOKE_HEIGHT", "218"))
TEST_H_ANGLE = int(os.environ.get("QWEN_SMOKE_H_ANGLE", "132"))
TEST_V_ANGLE = int(os.environ.get("QWEN_SMOKE_V_ANGLE", "41"))
TEST_ZOOM = float(os.environ.get("QWEN_SMOKE_ZOOM", "6.4"))


def select_option_by_hint(page: Page, selector: str, hint: str) -> str:
    return page.evaluate(
        """([selector, hint]) => {
          const select = document.querySelector(selector);
          if (!select) return '';
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


def setup_qwen(page: Page, workspace: str, prompt: str) -> dict[str, object]:
    page.evaluate(
        """([workspace, prompt, denoise, width, height]) => {
          setPreset(document.querySelector('[data-preset="Qwen"]'), 'Qwen');
          switchActivity('img2img', document.querySelector('[data-activity="img2img"]'));
          document.querySelector(workspace === 'multiview' ? '#tab-multiview' : (workspace === 'canvas' ? '#tab-canvas' : '#tab-viewer')).click();
          document.querySelector('#posPrompt').value = prompt;
          document.querySelector('#negPrompt').value = 'blur, artifacts, wrong color, identity drift, leak outside mask';
          document.querySelector('#widthInput').value = String(width);
          document.querySelector('#heightInput').value = String(height);
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#denoiseSlider').value = denoise;
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          updateSliderFromNumber('denoise');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }""",
        [workspace, prompt, TEST_DENOISE, TEST_WIDTH, TEST_HEIGHT],
    )
    page.wait_for_function(f"() => activePreset === 'Qwen' && activeWorkspace === '{workspace}'", timeout=60000)
    select_option_by_hint(page, "#modelSelect", "edit")
    select_option_by_hint(page, "#vaeSelect", "Qwen_Image")
    select_option_by_hint(page, "#textEncoderSelect", "qwen_2.5")
    select_option_by_hint(page, "#distilledLoraOneSelect", "Lightning-4steps")
    page.locator("#referenceImageInput").set_input_files(str(BASE_IMAGE))
    page.wait_for_function("() => !!referenceImageDataUrl && collectGenerationPayload()?.img2img?.reference_images?.length >= 1", timeout=60000)
    return page.evaluate("() => collectGenerationPayload()")


def paint_robe_mask(page: Page) -> None:
    page.evaluate(
        """() => {
          const sel = document.querySelector('#img2imgModeSelect');
          if (sel) sel.value = 'Inpaint masked area';
          const canvas = document.querySelector('#inpaintMaskCanvas');
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = 'rgba(255,255,255,1)';
          ctx.beginPath();
          ctx.ellipse(canvas.width * 0.50, canvas.height * 0.58, canvas.width * 0.22, canvas.height * 0.24, 0, 0, Math.PI * 2);
          ctx.fill();
          inpaintMaskDirty = true;
          updateWorkflowPreview();
        }"""
    )


def run_generation(page: Page, case: str) -> dict[str, object]:
    payload = page.evaluate("() => collectGenerationPayload()")
    if payload["width"] != TEST_WIDTH or payload["height"] != TEST_HEIGHT or payload["steps"] != 4 or float(payload["cfg"]) != 1.0:
        raise AssertionError(f"{case}: payload did not respect side menu: {payload!r}")
    job = page.evaluate(
        """async () => {
          const payload = collectGenerationPayload();
          const job = await startGenerationJob(payload);
          return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
        }"""
    )
    outputs = job.get("outputs") or []
    if not outputs:
        raise AssertionError(f"{case}: generation completed without outputs: {job!r}")
    output_path = local_output_path(outputs[0])
    metrics = analyze_image(output_path, case)
    image = Image.open(output_path)
    if image.size != (TEST_WIDTH, TEST_HEIGHT):
        raise AssertionError(f"{case}: output size {image.size} does not match requested {TEST_WIDTH}x{TEST_HEIGHT}")
    page.screenshot(path=str(RESULTS / f"{case}.png"), full_page=True)
    return {"case": case, "payload": payload, "job": job, "output": str(output_path), "metrics": metrics}


def pink_fraction(path: Path) -> float:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    roi = arr[int(h * 0.30): int(h * 0.86), int(w * 0.25): int(w * 0.75)]
    red, green, blue = roi[..., 0], roi[..., 1], roi[..., 2]
    return float(((red > 120) & (red > green * 1.25) & (blue > green * 0.65)).mean())


def main() -> None:
    if not BASE_IMAGE.exists():
        raise AssertionError(f"Missing smoke input: {BASE_IMAGE}")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)

        if "linear" in TEST_CASES:
            setup_qwen(page, "viewer", "change the green robe clothing to bright pink, keep face, pose and background unchanged")
            linear = run_generation(page, "qwen_linear_pink_real")
            if pink_fraction(Path(linear["output"])) < 0.02:
                raise AssertionError(f"qwen_linear_pink_real: pink edit is not visually detectable: {linear!r}")
            results.append(linear)

        if "multiview" in TEST_CASES:
            setup_qwen(page, "multiview", "preserve the same person and scene while changing the camera angle")
            page.evaluate(
                "([h, v, zoom]) => { setQwenMultiViewHorizontal(h); setQwenMultiViewVertical(v); setQwenMultiViewZoom(zoom); }",
                [TEST_H_ANGLE, TEST_V_ANGLE, TEST_ZOOM],
            )
            page.wait_for_function(
                "([h, v]) => collectGenerationPayload()?.video?.qwen_camera_horizontal === h && collectGenerationPayload()?.video?.qwen_camera_vertical === v",
                arg=[TEST_H_ANGLE, TEST_V_ANGLE],
                timeout=30000,
            )
            results.append(run_generation(page, "qwen_multiview_angle_real"))

        if "inpaint" in TEST_CASES:
            setup_qwen(page, "canvas", "change only the masked robe clothing to bright pink, keep unmasked face, hands and background unchanged")
            paint_robe_mask(page)
            payload = page.evaluate("() => collectGenerationPayload()")
            if not payload.get("img2img", {}).get("mask_image"):
                raise AssertionError("qwen_inpaint_pink_real: mask missing before generation")
            inpaint = run_generation(page, "qwen_inpaint_pink_real")
            if pink_fraction(Path(inpaint["output"])) < 0.025:
                raise AssertionError(f"qwen_inpaint_pink_real: masked pink edit is not visually detectable: {inpaint!r}")
            results.append(inpaint)
        browser.close()
    (RESULTS / "qwen_real_visual.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ok qwen real visual: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
