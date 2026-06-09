import json
from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
BASE_IMAGE = ROOT / "input" / "SmokeTest.png"
PRESETS = ["Qwen", "Flux", "Wan", "LTX", "SDXL"]


def main() -> None:
    if not BASE_IMAGE.exists():
        raise AssertionError(f"Missing smoke input: {BASE_IMAGE}")
    report = {}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)

        page.locator("button[data-preset='Qwen']").click()
        page.locator("[data-activity='img2img']").click()
        page.locator("#tab-canvas").click()
        page.locator("#referenceImageInput").set_input_files(str(BASE_IMAGE))
        page.wait_for_function("() => activeWorkspace === 'canvas' && !!referenceImageDataUrl", timeout=60000)
        page.evaluate(
            """() => {
              const sel = document.querySelector('#img2imgModeSelect');
              if (sel) sel.value = 'Inpaint masked area';
              const canvas = document.querySelector('#inpaintMaskCanvas');
              const ctx = canvas.getContext('2d');
              ctx.fillStyle = 'white';
              ctx.fillRect(20, 20, 80, 80);
              inpaintMaskDirty = true;
              updateWorkflowPreview();
            }"""
        )

        for preset in PRESETS:
            if page.locator(f"button[data-preset='{preset}']").count() == 0:
                continue
            page.locator(f"button[data-preset='{preset}']").click()
            page.locator("[data-activity='img2img']").click()
            page.locator("#tab-viewer").click()
            page.wait_for_function(f"() => activePreset === '{preset}' && activeWorkspace === 'viewer'", timeout=30000)
            payload = page.evaluate("() => collectGenerationPayload()")
            report[preset] = payload
            img2img = payload.get("img2img") or {}
            if img2img.get("mode") != "Image to Image":
                raise AssertionError(f"{preset}: linear viewer inherited inpaint mode: {payload!r}")
            if img2img.get("mask_image"):
                raise AssertionError(f"{preset}: linear viewer leaked an inpaint mask: {payload!r}")
        page.screenshot(path=str(RESULTS / "img2img-route-isolation-contract.png"), full_page=True)
        browser.close()
    (RESULTS / "img2img-route-isolation-contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ok img2img route isolation: linear viewer does not inherit canvas mask/mode")


if __name__ == "__main__":
    main()
