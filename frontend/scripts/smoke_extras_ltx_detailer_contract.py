import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
VIDEO = ROOT / "input" / "nexus_ltx_wan_motion_guide_4813b9fc9a.mp4"


def main() -> None:
    captured: dict[str, object] = {}

    def capture_extras(route: Route) -> None:
        form = route.request.post_data or ""
        marker = 'name="plan"'
        if marker not in form:
            raise AssertionError("Extras form did not include a plan field.")
        plan_text = form.split(marker, 1)[1].split("\r\n\r\n", 1)[1].split("\r\n--", 1)[0]
        captured["plan"] = json.loads(plan_text)
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": "extras-detailer-contract", "status": "queued", "progress": 0, "outputs": []}),
        )

    def poll_job(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": "extras-detailer-contract", "status": "completed", "progress": 100, "outputs": []}),
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.route("**/api/extras/start", capture_extras)
        page.route("**/api/extras/extras-detailer-contract", poll_job)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("#extrasVideoUpscaleEngine", state="attached", timeout=60000)
        page.evaluate(
            """
            () => {
              switchActivity('extras', document.querySelector('[data-activity="extras"]'));
              setExtrasMode('video', { quiet: true });
            }
            """
        )
        page.locator("#extrasFileInput").set_input_files(str(VIDEO))
        page.wait_for_function("() => typeof extrasSource !== 'undefined' && !!extrasSource && extrasSource.mediaType === 'video'", timeout=60000)
        page.evaluate(
            """
            () => {
              document.querySelector('#extrasVideoUpscaleToggle').checked = true;
              document.querySelector('#extrasVideoUpscaleEngine').value = 'ltx_detailer';
              document.querySelector('#extrasVideoUpscaleFactor').value = '2x';
              document.querySelector('#extrasVideoAlphaToggle').checked = false;
              document.querySelector('#extrasVideoEncodeToggle').checked = true;
              const encoder = document.querySelector('#extrasVideoEncoder');
              if (encoder) encoder.value = 'mp4_h264';
              syncExtrasVideoUpscaleUi();
              syncExtrasAlphaUi();
              syncExtrasEncodeUi();
            }
            """
        )
        page.locator("#extrasProcessBtn").click()
        page.wait_for_function("() => !document.querySelector('#extrasProcessingOverlay') || document.querySelector('#extrasProcessingOverlay').classList.contains('hidden')", timeout=60000)
        page.screenshot(path=str(RESULTS / "front_extras_ltx_detailer_contract.png"), full_page=True)
        browser.close()

    plan = captured.get("plan")
    if not isinstance(plan, dict):
        raise AssertionError("Extras plan was not captured.")
    upscale = plan.get("upscale")
    if not isinstance(upscale, dict):
        raise AssertionError(f"Missing upscale plan: {plan!r}")
    if upscale.get("engine") != "ltx_detailer" or not upscale.get("detailer_enabled"):
        raise AssertionError(f"Expected LTX detailer upscale plan, got {upscale!r}")
    if upscale.get("controlnet") != "Off" or upscale.get("image_bypass") is not True:
        raise AssertionError(f"Expected ControlNet Off + Image Bypass for detailer, got {upscale!r}")
    if plan.get("preserve_alpha"):
        raise AssertionError("Expected preserve_alpha disabled for MP4 test.")
    if plan.get("encoder") != "mp4_h264":
        raise AssertionError(f"Expected mp4_h264 encoder, got {plan.get('encoder')!r}")
    print("ok extras LTX detailer contract")


if __name__ == "__main__":
    main()
