import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from visual_checks import RESULTS, analyze_video, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
MOTION_GUIDE = ROOT / "input" / "nexus_ltx_wan_motion_guide_4813b9fc9a.mp4"
TARGET_IMAGE = ROOT / "input" / "Smoke_Character.png"
SECOND_IMAGE = ROOT / "input" / "nexus_smoke_reference2.jpg"


def main() -> None:
    for path in (MOTION_GUIDE, TARGET_IMAGE, SECOND_IMAGE):
        if not path.exists():
            raise FileNotFoundError(path)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 920})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.set_default_timeout(900000)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)
        page.evaluate(
            """() => {
              setPreset(document.querySelector('[data-preset="LTX"]'), 'LTX');
              switchActivity('img2img', document.querySelector('[data-activity="img2img"]'));
              switchView('director');
              document.querySelector('#widthInput').value = '512';
              document.querySelector('#heightInput').value = '512';
              document.querySelector('#stepsValue').value = '4';
              document.querySelector('#cfgValue').value = '1';
              document.querySelector('#ltxDirectorWidth').value = '512';
              document.querySelector('#ltxDirectorHeight').value = '512';
              document.querySelector('#ltxDirectorFps').value = '24';
              document.querySelector('#ltxDirectorDuration').value = '4';
              syncSlider('width');
              syncSlider('height');
              updateSliderFromNumber('steps');
              updateSliderFromNumber('cfg');
              syncLtxDirectorFromControls();
            }"""
        )
        page.locator("#ltxDirectorImageInput").set_input_files([str(TARGET_IMAGE), str(SECOND_IMAGE)])
        page.wait_for_function("() => (ltxDirectorState.segments || []).length >= 2", timeout=60000)
        page.evaluate(
            """() => {
              const [first, second] = ltxDirectorState.segments;
              first.start = 0;
              first.length = 2;
              first.prompt = 'Smoke character guided by reference motion, clean cinematic motion';
              first.negativePrompt = 'noise, artifacts, smear, frozen face';
              first.motionTransfer = { enabled: true, mode: 'pose', strength: 1 };
              second.start = 2;
              second.length = 1;
              second.prompt = 'Smoke character without motion transfer, calm hold';
              second.negativePrompt = 'noise, artifacts, smear';
              second.motionTransfer = { enabled: false, mode: 'pose', strength: 1 };
              ltxDirectorState.duration = 3;
              ltxDirectorState.width = 512;
              ltxDirectorState.height = 512;
              ltxDirectorState.fps = 24;
              ltxDirectorState.selectedType = 'segment';
              ltxDirectorState.selectedId = first.id;
              ltxDirectorPendingMotionSegmentId = first.id;
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }"""
        )
        page.locator("#ltxDirectorMotionVideoInput").set_input_files(str(MOTION_GUIDE))
        page.wait_for_function(
            """() => {
              const first = (ltxDirectorState.segments || [])[0];
              return !!first?.motionTransfer?.videoSrc && first.motionTransfer.enabled === true;
            }""",
            timeout=120000,
        )
        payload = page.evaluate("() => collectGenerationPayload()")
        director = payload.get("director") or {}
        timeline = director.get("timeline_data") or {}
        motion = timeline.get("motionTransfer") or []
        segments = timeline.get("segments") or []
        if payload.get("workspace") != "director" or payload.get("template") != "LTX_DIRECTOR_SUITE":
            raise AssertionError(f"Director payload did not route through legacy Director: {payload!r}")
        if payload.get("width") != 512 or payload.get("height") != 512 or payload.get("steps") != 4 or payload.get("cfg") != 1:
            raise AssertionError(f"Director payload did not keep 512x512 4 steps cfg 1: {payload!r}")
        if payload.get("video", {}).get("fps") != 24 or payload.get("video", {}).get("motion_transfer_enabled") is not True:
            raise AssertionError(f"Director motion transfer timing did not sync: {payload.get('video')!r}")
        if len(segments) < 2 or not segments[0].get("motionTransfer", {}).get("enabled") or segments[1].get("motionTransfer", {}).get("enabled"):
            raise AssertionError(f"Expected one motion segment and one plain segment: {segments!r}")
        if len(motion) != 1 or motion[0].get("mode") != "pose" or not str(motion[0].get("videoB64") or "").startswith("data:video/"):
            raise AssertionError(f"Director motion transfer metadata was not serialized: {motion!r}")

        job = page.evaluate(
            """async () => {
              const payload = collectGenerationPayload();
              const job = await startGenerationJob(payload);
              return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
            }"""
        )
        page.screenshot(path=str(RESULTS / "front_ltx_director_motion_transfer_real.png"), full_page=True)
        browser.close()

    outputs = [item for item in (job.get("outputs") or []) if str(item.get("path") or item.get("filename") or "").lower().endswith(".mp4")]
    if not outputs:
        raise AssertionError(f"Director motion transfer job completed without an mp4: {job!r}")
    path = local_output_path(outputs[0])
    metrics = analyze_video(path, "ltx_director_motion_transfer", frames=8, require_motion=True)
    result = {"payload": payload, "job": job, "output": str(path), "metrics": metrics}
    (RESULTS / "front_ltx_director_motion_transfer_real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"ok LTX Director motion transfer real battery: {path}")


if __name__ == "__main__":
    main()
