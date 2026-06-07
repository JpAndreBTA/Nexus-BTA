import json
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import Page, sync_playwright

from visual_checks import RESULTS, analyze_image, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
BASE_IMAGE = ROOT / "input" / "SmokeTeste_BaseMultipleReference.png"
REF2 = ROOT / "input" / "SmokeTeste_BaseMultipleReference2.png"


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


def setup_flux(page: Page, workspace: str, prompt: str) -> dict[str, object]:
    page.evaluate(
        """([workspace, prompt]) => {
          setPreset(document.querySelector('[data-preset="Flux"]'), 'Flux');
          switchActivity('img2img', document.querySelector('[data-activity="img2img"]'));
          document.querySelector(workspace === 'canvas' ? '#tab-canvas' : '#tab-viewer').click();
          document.querySelector('#posPrompt').value = prompt;
          document.querySelector('#negPrompt').value = 'blur, artifacts, wrong color, identity drift, leak outside mask';
          document.querySelector('#widthInput').value = '208';
          document.querySelector('#heightInput').value = '218';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#denoiseValue').value = '0.78';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          updateSliderFromNumber('denoise');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }""",
        [workspace, prompt],
    )
    page.wait_for_function(f"() => activePreset === 'Flux' && activeWorkspace === '{workspace}'", timeout=60000)
    select_option_by_hint(page, "#modelSelect", "Klein")
    select_option_by_hint(page, "#vaeSelect", "flux2-vae")
    select_option_by_hint(page, "#textEncoderSelect", "qwen_3_4b")
    page.locator("#referenceImageInput").set_input_files([str(BASE_IMAGE), str(REF2)])
    page.wait_for_function("() => !!referenceImageDataUrl && collectGenerationPayload()?.img2img?.reference_images?.length >= 2", timeout=60000)
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


def pink_fraction(path: Path) -> float:
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32)
    h, w, _ = arr.shape
    roi = arr[int(h * 0.30): int(h * 0.86), int(w * 0.25): int(w * 0.75)]
    red, green, blue = roi[..., 0], roi[..., 1], roi[..., 2]
    return float(((red > 120) & (red > green * 1.20) & (blue > green * 0.60)).mean())


def run_generation(page: Page, case: str) -> dict[str, object]:
    payload = page.evaluate("() => collectGenerationPayload()")
    if payload["width"] != 208 or payload["height"] != 218 or payload["steps"] != 4 or float(payload["cfg"]) != 1.0:
        raise AssertionError(f"{case}: payload did not respect side menu: {payload!r}")
    if len(payload.get("img2img", {}).get("reference_images") or []) < 2:
        raise AssertionError(f"{case}: multiple references missing from payload: {payload!r}")
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
    if image.size != (208, 218):
        raise AssertionError(f"{case}: output size {image.size} does not match requested 208x218")
    page.screenshot(path=str(RESULTS / f"{case}.png"), full_page=True)
    return {"case": case, "payload": payload, "job": job, "output": str(output_path), "metrics": metrics}


def main() -> None:
    for item in (BASE_IMAGE, REF2):
        if not item.exists():
            raise AssertionError(f"Missing smoke input: {item}")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)

        setup_flux(page, "viewer", "change the green robe clothing to bright pink, keep face, pose and background unchanged, use Image 2 as color/style reference")
        linear = run_generation(page, "flux2_linear_multiref_pink_real")
        results.append(linear)

        setup_flux(page, "canvas", "change only the masked robe clothing to bright pink, keep unmasked face, hands and background unchanged, use Image 2 as color/style reference")
        paint_robe_mask(page)
        payload = page.evaluate("() => collectGenerationPayload()")
        if not payload.get("img2img", {}).get("mask_image"):
            raise AssertionError("flux2_inpaint_multiref_pink_real: mask missing before generation")
        inpaint = run_generation(page, "flux2_inpaint_multiref_pink_real")
        results.append(inpaint)
        browser.close()
    (RESULTS / "flux2_real_visual.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ok flux2 real visual: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
