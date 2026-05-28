import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from visual_checks import RESULTS, analyze_image, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
SAMPLE = ROOT / "input" / "nexus_smoke_reference.png"


def main() -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)
        page.evaluate(
            """() => {
              setPreset(document.querySelector('[data-preset="Anima"]'), 'Anima');
              switchActivity('img2img', document.querySelector('[data-activity="img2img"]'));
              switchView('canvas');
              setInpaintMode('Inpaint masked area');
              document.querySelector('#widthInput').value = '512';
              document.querySelector('#heightInput').value = '512';
              document.querySelector('#stepsValue').value = '4';
              document.querySelector('#cfgValue').value = '1';
              document.querySelector('#inpaintEngineSelect').value = 'lanpaint';
              document.querySelector('#lanpaintThinkingStepsInput').value = '5';
              syncSlider('width');
              syncSlider('height');
              updateSliderFromNumber('steps');
              updateSliderFromNumber('cfg');
              syncInpaintEngineControls();
            }"""
        )
        page.locator("#referenceImageInput").set_input_files(str(SAMPLE))
        page.wait_for_function("() => !!referenceImageDataUrl && activeWorkspace === 'canvas'", timeout=60000)
        canvas = page.locator("#inpaintMaskCanvas")
        canvas.wait_for(state="attached", timeout=30000)
        page.wait_for_function("() => document.querySelector('#inpaintMaskCanvas')?.width > 16", timeout=30000)
        box = canvas.bounding_box()
        if not box:
            raise AssertionError("Legacy inpaint canvas did not expose a bounding box.")

        def stroke(tool: str, rx: float, ry: float, dx: float = 26, dy: float = 0) -> None:
            page.evaluate("(tool) => setInpaintMode(tool)", tool)
            live = canvas.bounding_box()
            if not live:
                raise AssertionError("Legacy inpaint canvas lost its bounding box.")
            x = live["x"] + live["width"] * rx
            y = live["y"] + live["height"] * ry
            page.mouse.move(x, y)
            page.mouse.down()
            page.mouse.move(x + dx, y + dy)
            page.mouse.up()

        def click_tool(tool: str, rx: float, ry: float, paint_mode: str = "fill") -> dict:
            page.evaluate(
                """([tool, paintMode]) => {
                  setInpaintMode(tool);
                  document.querySelector('#inpaintWandPaintMode').value = paintMode;
                }""",
                [tool, paint_mode],
            )
            live = canvas.bounding_box()
            if not live:
                raise AssertionError("Legacy inpaint canvas lost its bounding box.")
            page.mouse.click(live["x"] + live["width"] * rx, live["y"] + live["height"] * ry)
            page.wait_for_timeout(250)
            return page.evaluate("() => inpaintMaskIntentSummary()")

        stroke("Inpaint masked area", 0.48, 0.48)
        stroke("Remove mask", 0.63, 0.48)
        wand_intent = click_tool("Magic Wand", 0.32, 0.36, "fill")
        page.evaluate("() => undoInpaintMask()")
        page.wait_for_timeout(150)
        page.evaluate("() => redoInpaintMask()")
        page.wait_for_timeout(150)
        object_intent = click_tool("Select Object", 0.50, 0.46, "remove")
        if wand_intent.get("fill_pixels", 0) <= 0:
            raise AssertionError(f"Magic Wand did not paint a fill mask: {wand_intent!r}")
        if object_intent.get("remove_pixels", 0) <= 0:
            raise AssertionError(f"Select Object did not paint a remove mask: {object_intent!r}")
        page.evaluate("() => extendCanvasArea('top', { top: 128 })")
        page.wait_for_function("() => document.querySelector('#inpaintMaskCanvas')?.height === 640", timeout=30000)
        page.evaluate("() => undoInpaintMask()")
        page.wait_for_function("() => document.querySelector('#inpaintMaskCanvas')?.height === 512", timeout=30000)
        page.evaluate("() => redoInpaintMask()")
        page.wait_for_function("() => document.querySelector('#inpaintMaskCanvas')?.height === 640", timeout=30000)
        canvas_state = page.evaluate(
            """() => ({
              width: inpaintMaskCanvas.width,
              height: inpaintMaskCanvas.height,
              sideWidth: widthInput.value,
              sideHeight: heightInput.value,
              label: inpaintAreaLabel.innerText,
              intent: inpaintMaskIntentSummary(),
              hasMask: !!exportInpaintMaskDataUrl()
            })"""
        )
        if canvas_state["width"] != 512 or canvas_state["height"] != 640:
            raise AssertionError(f"Generative Fill did not update canvas dimensions: {canvas_state!r}")
        if canvas_state["sideWidth"] != "512" or canvas_state["sideHeight"] != "640":
            raise AssertionError(f"Generative Fill did not sync side-menu dimensions: {canvas_state!r}")
        if canvas_state["intent"].get("intent") != "mixed":
            raise AssertionError(f"Remove Mask + fill intent was not detected: {canvas_state!r}")
        if not canvas_state["hasMask"]:
            raise AssertionError(f"Unified inpaint mask was not exportable: {canvas_state!r}")
        payload = page.evaluate("() => collectGenerationPayload()")
        if payload["img2img"].get("inpaint_engine") != "lanpaint":
            raise AssertionError(f"Legacy inpaint payload did not default to LanPaint: {payload['img2img']!r}")
        if payload["width"] != 512 or payload["height"] != 640:
            raise AssertionError(f"Payload dimensions did not follow Generative Fill: {payload['width']}x{payload['height']}")
        if payload["img2img"].get("inpaint_intent") != "mixed":
            raise AssertionError(f"Payload did not include mixed inpaint intent: {payload['img2img']!r}")
        job = page.evaluate(
            """async () => {
              const payload = collectGenerationPayload();
              const job = await startGenerationJob(payload);
              return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
            }"""
        )
        page.screenshot(path=str(RESULTS / "front_legacy_inpaint_lanpaint_real.png"), full_page=True)
        browser.close()

    outputs = job.get("outputs") or []
    if not outputs:
        raise AssertionError(f"Legacy inpaint job completed without outputs: {job!r}")
    path = local_output_path(outputs[0])
    metrics = analyze_image(path, "legacy_inpaint_lanpaint")
    result = {"payload": payload, "canvas_state": canvas_state, "job": job, "output": str(path), "metrics": metrics}
    (RESULTS / "front_legacy_inpaint_lanpaint_real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ok legacy inpaint LanPaint real battery: {path}")


if __name__ == "__main__":
    main()
