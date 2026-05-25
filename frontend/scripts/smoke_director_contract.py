import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/app"
SAMPLE = ROOT / "temp" / "extras_validation" / "sample.png"
ALPHA = ROOT / "temp" / "extras_validation" / "alpha_source.png"
THIRD = ROOT / "output" / "extras" / "image" / "20260525_054329_upscale_5681d9.png"


def job_body() -> str:
    return json.dumps(
        {
            "job_id": "director-contract-smoke",
            "prompt_id": None,
            "status": "completed",
            "progress": 100,
            "message": "director contract smoke",
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
        page.route("**/api/generate/director-contract-smoke", lambda route: route.fulfill(status=200, content_type="application/json", body=job_body()))

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.get_by_role("button", name="LTX 2.3").click()
        page.get_by_role("button", name="img2img").click()
        page.get_by_placeholder("Describe the image...").fill("ltx director contract")
        page.locator(".img2img-source input[type='file']").first.set_input_files(str(SAMPLE))
        extras = [str(ALPHA)]
        if THIRD.exists():
            extras.append(str(THIRD))
        page.locator(".multi-reference-dropzone input[type='file']").set_input_files(extras)
        page.get_by_label("Toggle LTX Director").click()
        page.get_by_role("button", name="Generate").click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(RESULTS / "app-smoke-director-contract.png"), full_page=True)
        browser.close()

    if len(payloads) != 1:
        raise AssertionError(f"Expected 1 payload, captured {len(payloads)}.")
    payload = payloads[0]
    if payload.get("preset") != "LTX" or payload.get("workspace") != "director" or payload.get("activity") != "txt2img":
        raise AssertionError(f"Unexpected Director route fields: {payload!r}")
    if payload.get("template") != "LTX_DIRECTOR_SUITE":
        raise AssertionError(f"Missing Director template: {payload!r}")

    img2img = payload.get("img2img")
    if not isinstance(img2img, dict):
        raise AssertionError(f"Missing img2img settings: {payload!r}")
    refs = img2img.get("reference_images")
    if not isinstance(refs, list) or len(refs) != 3:
        raise AssertionError(f"Expected 3 capped references, got {len(refs) if isinstance(refs, list) else type(refs).__name__}.")
    if any(not str(ref).startswith("data:image/") for ref in refs):
        raise AssertionError("Expected all references to be image data URLs.")

    director = payload.get("director")
    if not isinstance(director, dict):
        raise AssertionError(f"Missing Director payload: {payload!r}")
    if director.get("frame_rate") != 24 or director.get("duration_seconds") != 12:
        raise AssertionError(f"Unexpected Director timing: {director!r}")
    if not isinstance(director.get("timeline_data_json"), str):
        raise AssertionError("Expected timeline_data_json string.")
    json.loads(str(director["timeline_data_json"]))

    video = payload.get("video")
    if not isinstance(video, dict) or not video.get("director_timeline"):
        raise AssertionError(f"Expected video.director_timeline, got {video!r}.")

    print("ok director contract: LTX director + multi references")


if __name__ == "__main__":
    main()
