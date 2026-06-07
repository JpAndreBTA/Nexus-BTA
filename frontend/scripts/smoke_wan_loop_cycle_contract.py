import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
SAMPLE = ROOT / "input" / "Smoke_splashART.jpeg"


def job_body(job_id: str) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "prompt_id": None,
            "status": "completed",
            "progress": 100,
            "message": "wan loop cycle contract",
            "outputs": [],
            "error": None,
            "created_at": "2026-05-31T00:00:00",
            "updated_at": "2026-05-31T00:00:00",
        }
    )


def set_wan_base(page) -> None:
    page.locator("button[data-preset='Wan']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'Wan' && currentActivity === 'img2img'", timeout=60000)
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncWanLoopCycleToggle === 'function') syncWanLoopCycleToggle(false);
          document.querySelector('#widthInput').value = '928';
          document.querySelector('#heightInput').value = '480';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#fpsInput').value = '24';
          document.querySelector('#secondsInput').value = '5';
          document.querySelector('#framesInput').value = '121';
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Simple';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          syncVideoMotionFields('framesInput');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
    )
    page.locator("#referenceImageInput").set_input_files(str(SAMPLE))
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'Wan'
            && payload?.activity === 'img2img'
            && payload?.img2img?.reference_images?.length === 1
            && payload?.video?.wan_loop_cycle === false;
        }""",
        timeout=60000,
    )


def click_generate(page) -> None:
    page.locator("#globalGenerateButton").click()
    page.wait_for_timeout(250)


def main() -> None:
    if not SAMPLE.exists():
        raise AssertionError(f"Missing smoke input: {SAMPLE}")
    payloads: list[dict[str, object]] = []

    def capture_generate(route: Route) -> None:
        payloads.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body=job_body(f"wan-loop-contract-{len(payloads)}"))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.route("**/api/generate/start", capture_generate)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => typeof collectGenerationPayload === 'function'", timeout=120000)

        set_wan_base(page)
        click_generate(page)

        page.evaluate("() => syncWanLoopCycleToggle(true)")
        page.wait_for_function("() => collectGenerationPayload()?.video?.wan_loop_cycle === true", timeout=30000)
        click_generate(page)

        set_wan_base(page)
        page.locator("#referenceImageInput").set_input_files([str(SAMPLE), str(SAMPLE)])
        page.wait_for_function(
            """() => {
              const payload = collectGenerationPayload();
              return payload?.img2img?.reference_images?.length === 2
                && payload?.video?.wan_loop_cycle === false;
            }""",
            timeout=60000,
        )
        click_generate(page)
        page.screenshot(path=str(RESULTS / "wan-loop-cycle-contract.png"), full_page=True)
        browser.close()

    if len(payloads) != 3:
        raise AssertionError(f"Expected 3 payloads, captured {len(payloads)}")

    normal_start, loop_start, normal_start_end = payloads
    for case, payload in {
        "normal_start": normal_start,
        "loop_start": loop_start,
        "normal_start_end": normal_start_end,
    }.items():
        video = payload.get("video")
        if payload.get("preset") != "Wan" or payload.get("activity") != "img2img" or not isinstance(video, dict):
            raise AssertionError(f"{case}: unexpected payload route: {payload!r}")
        if payload.get("workflow_override") is not None or payload.get("workflow_id") is not None:
            raise AssertionError(f"{case}: Wan loop battery must use backend workflow builder, got override/id: {payload!r}")
        if payload.get("width") != 928 or payload.get("height") != 480:
            raise AssertionError(f"{case}: dimensions not synced to 480p aspect: {payload!r}")
        if video.get("fps") != 24 or video.get("seconds") != 5 or video.get("frames") != 121:
            raise AssertionError(f"{case}: video timing not synced: {video!r}")

    if normal_start["video"].get("wan_loop_cycle") is not False:
        raise AssertionError(f"normal start was contaminated by loop flag: {normal_start!r}")
    if loop_start["video"].get("wan_loop_cycle") is not True or loop_start["video"].get("wan_loop_source") != "start_frame_as_end_frame":
        raise AssertionError(f"loop start did not send loop source: {loop_start!r}")
    if normal_start_end["video"].get("wan_loop_cycle") is not False:
        raise AssertionError(f"normal start+end was contaminated by loop flag: {normal_start_end!r}")
    if len((normal_start_end.get("img2img") or {}).get("reference_images") or []) != 2:
        raise AssertionError(f"normal start+end lost the end reference: {normal_start_end!r}")

    print("ok wan loop cycle contract: normal start/start+end isolated, loop flag synced")


if __name__ == "__main__":
    main()
