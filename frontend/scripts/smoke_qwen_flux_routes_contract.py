import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
BASE_IMAGE = ROOT / "input" / "SmokeTeste_BaseMultipleReference.png"
REF2 = ROOT / "input" / "SmokeTeste_BaseMultipleReference2.png"


def job_body(job_id: str) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "prompt_id": None,
            "status": "completed",
            "progress": 100,
            "message": "route contract",
            "outputs": [],
            "error": None,
            "created_at": "2026-06-01T00:00:00",
            "updated_at": "2026-06-01T00:00:00",
        }
    )


def set_side_menu(page, *, prompt: str) -> None:
    page.evaluate(
        """prompt => {
          clearReferenceImage({ quiet: true });
          document.querySelector('#posPrompt').value = prompt;
          document.querySelector('#negPrompt').value = 'blur, leak outside mask, wrong color';
          document.querySelector('#widthInput').value = '208';
          document.querySelector('#heightInput').value = '218';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }""",
        prompt,
    )


def select_qwen_lightning_lora(page) -> str:
    return page.evaluate(
        """() => {
          const select = document.querySelector('#distilledLoraOneSelect');
          if (!select) return '';
          const option = [...select.options].find(item => /qwen.*lightning/i.test(`${item.value} ${item.textContent}`));
          if (!option) return '';
          select.value = option.value;
          select.dispatchEvent(new Event('change', { bubbles: true }));
          const strength = document.querySelector('#distilledLoraOneStrength');
          if (strength) strength.value = '1';
          updateWorkflowPreview();
          return option.value;
        }"""
    )


def select_route(page, preset: str, workspace: str, *, prompt: str) -> None:
    page.locator(f"button[data-preset='{preset}']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function(f"() => activePreset === '{preset}' && currentActivity === 'img2img'", timeout=60000)
    if workspace == "multiview":
        page.locator("#tab-multiview").click()
        page.wait_for_function("() => activeWorkspace === 'multiview'", timeout=30000)
    elif workspace == "canvas":
        page.locator("#tab-canvas").click()
        page.wait_for_function("() => activeWorkspace === 'canvas'", timeout=30000)
    else:
        page.locator("#tab-viewer").click()
        page.wait_for_function("() => activeWorkspace === 'viewer'", timeout=30000)
    set_side_menu(page, prompt=prompt)
    page.locator("#referenceImageInput").set_input_files(str(BASE_IMAGE))
    page.wait_for_function("() => !!referenceImageDataUrl && collectGenerationPayload()?.img2img?.reference_images?.length >= 1", timeout=60000)


def paint_mask(page) -> None:
    page.evaluate(
        """() => {
          const sel = document.querySelector('#img2imgModeSelect');
          if (sel) sel.value = 'Inpaint masked area';
          const canvas = document.querySelector('#inpaintMaskCanvas');
          const ctx = canvas.getContext('2d');
          ctx.clearRect(0, 0, canvas.width, canvas.height);
          ctx.fillStyle = 'rgba(255,255,255,1)';
          ctx.fillRect(Math.floor(canvas.width * 0.35), Math.floor(canvas.height * 0.25), Math.floor(canvas.width * 0.25), Math.floor(canvas.height * 0.35));
          inpaintMaskDirty = true;
          updateWorkflowPreview();
        }"""
    )


def click_generate(page) -> None:
    page.locator("#globalGenerateButton").click()
    page.wait_for_timeout(250)


def click_multiview_generate(page) -> None:
    page.locator("#view-multiview button", has_text="Generate MultiView").click()
    page.wait_for_timeout(250)


def main() -> None:
    for item in (BASE_IMAGE, REF2):
        if not item.exists():
            raise AssertionError(f"Missing smoke input: {item}")
    captured: list[dict[str, object]] = []

    def capture(route: Route) -> None:
        captured.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body=job_body(f"routes-{len(captured)}"))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.route("**/api/qwen/multiview/status", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"installed": True, "name": "qwen\\qwen-image-edit-2511-multiple-angles-lora.safetensors", "size_bytes": 295140688})))
        page.route("**/api/generate/start", capture)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => typeof collectGenerationPayload === 'function' && backendOnline === true", timeout=120000)

        select_route(page, "Qwen", "viewer", prompt="change the dress clothing color to pink, keep the person and background unchanged")
        qwen_lightning = select_qwen_lightning_lora(page)
        if not qwen_lightning:
            raise AssertionError("qwen_linear: no Qwen Lightning LoRA option found in side menu")
        click_generate(page)

        select_route(page, "Qwen", "multiview", prompt="generate a new camera angle while preserving the same subject")
        qwen_multiview_lightning = select_qwen_lightning_lora(page)
        if not qwen_multiview_lightning:
            raise AssertionError("qwen_multiview: no Qwen Lightning LoRA option found in side menu")
        page.evaluate("() => { setQwenMultiViewHorizontal(90); setQwenMultiViewVertical(0); setQwenMultiViewZoom(6.4); }")
        page.wait_for_function(
            "() => collectGenerationPayload()?.video?.qwen_camera_horizontal === 90 && /right side view/i.test(collectGenerationPayload()?.video?.qwen_camera_prompt || '')",
            timeout=30000,
        )
        page.evaluate("() => setQwenMultiViewHorizontal(270)")
        page.wait_for_function(
            "() => collectGenerationPayload()?.video?.qwen_camera_horizontal === 270 && /left side view/i.test(collectGenerationPayload()?.video?.qwen_camera_prompt || '')",
            timeout=30000,
        )
        page.evaluate("() => { setQwenMultiViewHorizontal(132); setQwenMultiViewVertical(41); setQwenMultiViewZoom(6.4); }")
        page.wait_for_function("() => collectGenerationPayload()?.video?.qwen_camera_horizontal === 132 && collectGenerationPayload()?.video?.qwen_camera_vertical === 41", timeout=30000)
        page.wait_for_function("() => document.querySelector('#posPrompt')?.readOnly === true && document.querySelector('#negPrompt')?.readOnly === true", timeout=30000)
        click_generate(page)
        click_multiview_generate(page)

        select_route(page, "Qwen", "canvas", prompt="change only the masked clothing to pink, keep unmasked pixels unchanged")
        paint_mask(page)
        click_generate(page)

        select_route(page, "Flux", "viewer", prompt="change the dress clothing color to pink, keep the person and background unchanged")
        click_generate(page)

        select_route(page, "Flux", "canvas", prompt="change only the masked clothing to pink, keep unmasked pixels unchanged")
        paint_mask(page)
        click_generate(page)

        page.screenshot(path=str(RESULTS / "qwen-flux-routes-contract.png"), full_page=True)
        browser.close()

    if len(captured) != 6:
        raise AssertionError(f"Expected 6 captured payloads, got {len(captured)}")
    names = ["qwen_linear", "qwen_multiview", "qwen_multiview_button", "qwen_inpaint", "flux_linear", "flux_inpaint"]
    report = dict(zip(names, captured))
    for name, payload in report.items():
        if payload.get("width") != 208 or payload.get("height") != 218:
            raise AssertionError(f"{name}: side menu size leaked: {payload!r}")
        if payload.get("steps") != 4 or float(payload.get("cfg") or 0) != 1.0:
            raise AssertionError(f"{name}: steps/cfg did not follow side menu: {payload!r}")
        if payload.get("workflow_id") is not None or payload.get("workflow_override") is not None:
            raise AssertionError(f"{name}: inherited workflow route unexpectedly: {payload!r}")
        if name not in {"qwen_multiview", "qwen_multiview_button"} and "pink" not in str(payload.get("prompt", "")).lower():
            raise AssertionError(f"{name}: edit prompt was not preserved: {payload!r}")
    for qwen_name in ("qwen_linear", "qwen_multiview", "qwen_multiview_button"):
        distilled = report[qwen_name].get("distilled_loras") or []
        if not any("qwen" in str(item.get("name", "")).lower() and "lightning" in str(item.get("name", "")).lower() for item in distilled if isinstance(item, dict)):
            raise AssertionError(f"{qwen_name}: selected Qwen Lightning LoRA missing from payload: {report[qwen_name]!r}")
    for mv_name in ("qwen_multiview", "qwen_multiview_button"):
        mv = report[mv_name]
        if mv.get("workspace") != "multiview" or mv.get("video", {}).get("qwen_multiview") is not True:
            raise AssertionError(f"{mv_name}: not routed as multiview: {mv!r}")
        if mv.get("video", {}).get("qwen_camera_horizontal") != 132 or mv.get("video", {}).get("qwen_camera_vertical") != 41:
            raise AssertionError(f"{mv_name}: side camera angles not synced: {mv!r}")
        if float(mv.get("video", {}).get("qwen_camera_zoom") or 0) != 6.4:
            raise AssertionError(f"{mv_name}: camera zoom not synced: {mv!r}")
        if "<sks>" not in str(mv.get("video", {}).get("qwen_camera_prompt", "")):
            raise AssertionError(f"{mv_name}: camera prompt missing from payload: {mv!r}")
        if mv.get("prompt") != mv.get("video", {}).get("qwen_camera_prompt"):
            raise AssertionError(f"{mv_name}: free prompt leaked instead of camera prompt: {mv!r}")
        if mv.get("negative_prompt"):
            raise AssertionError(f"{mv_name}: negative prompt should be disabled: {mv!r}")
        if abs(float(mv.get("denoise") or 0) - 1.0) > 0.001 or abs(float(mv.get("img2img", {}).get("denoise") or 0) - 1.0) > 0.001:
            raise AssertionError(f"{mv_name}: multiview must use full reference denoise: {mv!r}")
        if mv.get("video", {}).get("qwen_auto_edit_lora") is not True:
            raise AssertionError(f"{mv_name}: qwen edit lightning auto flag missing: {mv!r}")
    for name in ("qwen_inpaint", "flux_inpaint"):
        if report[name].get("workspace") != "canvas" or not report[name].get("img2img", {}).get("mask_image"):
            raise AssertionError(f"{name}: inpaint mask/workspace missing: {report[name]!r}")
    (RESULTS / "qwen-flux-routes-contract.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("ok qwen/flux routes contract: linear, multiview and inpaint respect side menu and route isolation")


if __name__ == "__main__":
    main()
