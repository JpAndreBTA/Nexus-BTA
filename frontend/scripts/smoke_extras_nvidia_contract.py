import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
IMAGE = ROOT / "frontend" / "results" / "manual_compare" / "smoke_endframe_thumb.jpg"
VIDEO = ROOT / "output" / "video" / "20260531_202058_Wan_i2v_NEXUS_BTA_WAN22_LOOP_CYCLE_00001_.mp4"


def plan_from_form(route: Route) -> dict[str, object]:
    form = route.request.post_data or ""
    marker = 'name="plan"'
    if marker not in form:
        raise AssertionError("Extras form did not include a plan field.")
    plan_text = form.split(marker, 1)[1].split("\r\n\r\n", 1)[1].split("\r\n--", 1)[0]
    return json.loads(plan_text)


def main() -> None:
    captured: list[dict[str, object]] = []

    def capture_extras(route: Route) -> None:
        captured.append(plan_from_form(route))
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": f"extras-nvidia-contract-{len(captured)}", "status": "queued", "progress": 0, "outputs": []}),
        )

    def poll_job(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"job_id": "extras-nvidia-contract", "status": "completed", "progress": 100, "outputs": []}),
        )

    def status(route: Route) -> None:
        engine = "nvidia_pid" if "nvidia_pid" in route.request.url else "nvidia_rtx"
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "engine": engine,
                    "label": "NVIDIA PiD Pixel Diffusion Decoder" if engine == "nvidia_pid" else "NVIDIA RTX Video Super Resolution",
                    "installed": True,
                    "node_ready": True,
                    "dependency_ready": True,
                    "expected_nodes": ["ComfyUI-PiD"] if engine == "nvidia_pid" else ["Nvidia_RTX_Nodes_ComfyUI"],
                    "packages": {},
                    "model_required": engine == "nvidia_pid",
                    "models_auto_download": engine == "nvidia_pid",
                }
            ),
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.on("dialog", lambda dialog: dialog.accept())
        page.route("**/api/extras/start", capture_extras)
        page.route("**/api/extras/extras-nvidia-contract-*", poll_job)
        page.route("**/api/extras/nvidia/*/status", status)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_selector("#extrasImageUpscaler", state="attached", timeout=60000)

        page.evaluate("() => { switchActivity('extras', document.querySelector('[data-activity=\"extras\"]')); setExtrasMode('image', { quiet: true }); }")
        page.locator("#extrasFileInput").set_input_files(str(IMAGE))
        page.wait_for_function("() => typeof extrasSource !== 'undefined' && !!extrasSource && extrasSource.mediaType === 'image'", timeout=60000)
        page.select_option("#extrasImageUpscaler", "nvidia_pid")
        page.locator("#extrasProcessBtn").click()
        page.wait_for_timeout(500)

        page.evaluate("() => { setExtrasMode('video', { quiet: true }); }")
        page.locator("#extrasFileInput").set_input_files(str(VIDEO))
        page.wait_for_function("() => typeof extrasSource !== 'undefined' && !!extrasSource && extrasSource.mediaType === 'video'", timeout=60000)
        page.evaluate(
            """
            () => {
              document.querySelector('#extrasVideoUpscaleToggle').checked = true;
              document.querySelector('#extrasVideoUpscaleEngine').value = 'nvidia_rtx';
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
        page.wait_for_timeout(500)
        page.screenshot(path=str(RESULTS / "front_extras_nvidia_contract.png"), full_page=True)
        browser.close()

    if len(captured) != 2:
        raise AssertionError(f"Expected two Extras plans, captured {len(captured)}")
    image_upscale = captured[0].get("upscale")
    video_upscale = captured[1].get("upscale")
    if not isinstance(image_upscale, dict) or image_upscale.get("engine") != "nvidia_pid":
        raise AssertionError(f"Expected image PiD plan, got {captured[0]!r}")
    if not isinstance(video_upscale, dict) or video_upscale.get("engine") != "nvidia_rtx":
        raise AssertionError(f"Expected video RTX plan, got {captured[1]!r}")
    if captured[1].get("encoder") != "mp4_h264" or captured[1].get("preserve_alpha"):
        raise AssertionError(f"Expected mp4_h264 without alpha, got {captured[1]!r}")
    (RESULTS / "front_extras_nvidia_contract.json").write_text(json.dumps(captured, indent=2), encoding="utf-8")
    print("ok extras NVIDIA contract")


if __name__ == "__main__":
    main()
