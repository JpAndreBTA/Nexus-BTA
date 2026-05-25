import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/app"
SAMPLE = ROOT / "temp" / "extras_validation" / "sample.png"


def job_body() -> str:
    return json.dumps(
        {
            "job_id": "video-contract-smoke",
            "prompt_id": None,
            "status": "completed",
            "progress": 100,
            "message": "video contract smoke",
            "outputs": [],
            "error": None,
            "created_at": "2026-05-25T00:00:00",
            "updated_at": "2026-05-25T00:00:00",
        }
    )


def main() -> None:
    payloads: list[dict[str, object]] = []

    def capture_generate(route: Route) -> None:
        payloads.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body=job_body())

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.route("**/api/generate/start", capture_generate)
        page.route("**/api/generate/video-contract-smoke", lambda route: route.fulfill(status=200, content_type="application/json", body=job_body()))

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        preset = page.locator("label.field:has-text('Preset') select")
        model = page.locator("label.field:has-text('Model') select")

        preset.select_option(label="LTX")
        model.wait_for(timeout=30000)
        page.get_by_placeholder("Describe the image...").fill("ltx video contract")
        page.get_by_role("button", name="Generate").click()
        page.wait_for_timeout(500)

        preset.select_option(label="Wan")
        page.get_by_role("button", name="img2img").click()
        page.locator(".img2img-source input[type='file']").first.set_input_files(str(SAMPLE))
        page.get_by_placeholder("Describe the image...").fill("wan video contract")
        page.get_by_role("button", name="Generate").click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(RESULTS / "app-smoke-video-contract.png"), full_page=True)
        browser.close()

    if len(payloads) != 2:
        raise AssertionError(f"Expected 2 video payloads, captured {len(payloads)}.")

    ltx, wan = payloads
    ltx_video = ltx.get("video")
    wan_video = wan.get("video")
    if not isinstance(ltx_video, dict) or ltx.get("preset") != "LTX":
        raise AssertionError(f"Invalid LTX payload: {ltx!r}")
    if ltx.get("steps") != 8 or ltx.get("cfg") != 1 or ltx.get("sampler") != "euler_ancestral_cfg_pp":
        raise AssertionError(f"Unexpected LTX generation defaults: {ltx!r}")
    if ltx_video.get("frames") != 121 or ltx_video.get("fps") != 24 or ltx_video.get("seconds") != 5:
        raise AssertionError(f"Unexpected LTX video defaults: {ltx_video!r}")
    if ltx_video.get("decode_tiles_x") != 2 or ltx_video.get("decode_tiles_y") != 2:
        raise AssertionError(f"Unexpected LTX decode defaults: {ltx_video!r}")

    if not isinstance(wan_video, dict) or wan.get("preset") != "Wan":
        raise AssertionError(f"Invalid Wan payload: {wan!r}")
    if wan.get("activity") != "img2img":
        raise AssertionError(f"Wan contract should use img2img in this smoke: {wan!r}")
    if wan.get("steps") != 4 or wan.get("cfg") != 1 or wan.get("sampler") != "euler":
        raise AssertionError(f"Unexpected Wan generation defaults: {wan!r}")
    if wan_video.get("frames") != 81 or wan_video.get("fps") != 16 or wan_video.get("seconds") != 5:
        raise AssertionError(f"Unexpected Wan video defaults: {wan_video!r}")

    print("ok video contract: LTX + Wan defaults")


if __name__ == "__main__":
    main()
