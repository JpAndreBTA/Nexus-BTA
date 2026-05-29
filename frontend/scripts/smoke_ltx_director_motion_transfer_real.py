import json
from pathlib import Path

from playwright.sync_api import sync_playwright

from visual_checks import RESULTS, analyze_video, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
MOTION_GUIDE = ROOT / "input" / "nexus_ltx_wan_motion_guide_56cae91f3b.mp4"
CAMERA_MOTION_GUIDE = ROOT / "input" / "CameraMan_ref.mp4"
TARGET_IMAGE = ROOT / "input" / "Smoke_Character.png"
START_REFERENCE = ROOT / "input" / "nexus_smoke_reference.png"
END_REFERENCE = ROOT / "input" / "nexus_smoke_reference2.jpg"
CAMERA_END_FRAME = ROOT / "input" / "Smoke_EndFrame.png"


def select_option_by_hint(page, selector: str, hint: str, required: bool = False) -> str:
    value = page.evaluate(
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
    if required and not value:
        raise AssertionError(f"Could not select {selector} by hint {hint!r}")
    return value


def main() -> None:
    for path in (MOTION_GUIDE, CAMERA_MOTION_GUIDE, TARGET_IMAGE, START_REFERENCE, END_REFERENCE, CAMERA_END_FRAME):
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
        def reset_director() -> None:
            page.evaluate(
                """() => {
              ltxDirectorState.segments = [];
              ltxDirectorState.referenceImages = [];
              ltxDirectorState.audioSegments = [];
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
              if (document.querySelector('#latentUpscaleSelect') && /^(automatic|auto|none|)$/i.test(document.querySelector('#latentUpscaleSelect').value || 'Automatic')) {
                document.querySelector('#latentUpscaleSelect').value = 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors';
              }
              if (document.querySelector('#ltxLatentUpscaleRefineToggle')) document.querySelector('#ltxLatentUpscaleRefineToggle').checked = true;
              if (document.querySelector('#ltxDetailerToggle')) document.querySelector('#ltxDetailerToggle').checked = true;
              if (document.querySelector('#ltxDetailerLoraSelect')) {
                const detailer = [...document.querySelector('#ltxDetailerLoraSelect').options].find(option => /detailer/i.test(option.value + option.textContent));
                if (detailer) document.querySelector('#ltxDetailerLoraSelect').value = detailer.value;
              }
              renderLtxDirector();
            }"""
            )
            select_option_by_hint(page, "#modelSelect", "Dasiwa", required=False)

        def configure_motion_plain_case(mode: str = "pose", guide_path: Path = MOTION_GUIDE, case_name: str = "motion_plain") -> dict:
            reset_director()
            page.locator("#ltxDirectorImageInput").set_input_files([str(TARGET_IMAGE), str(START_REFERENCE)])
            page.wait_for_function("() => (ltxDirectorState.segments || []).length >= 2", timeout=60000)
            page.evaluate(
                """mode => {
              const [first, second] = ltxDirectorState.segments;
              first.start = 0;
              first.length = 2;
              first.prompt = mode === 'camera'
                ? 'same target image subject, same face, same hair, same polka dot outfit, camera orbit from reference video'
                : 'Smoke character guided by reference motion, clean cinematic motion';
              first.negativePrompt = mode === 'camera'
                ? 'different person, changed face, changed gender, identity drift, noise, artifacts, smear'
                : 'noise, artifacts, smear, frozen face';
              first.motionTransfer = { enabled: true, mode, strength: 1 };
              second.start = 2.25;
              second.length = 3.25;
              second.prompt = 'Plain reference segment without motion transfer, stable clear identity';
              second.negativePrompt = 'noise, artifacts, smear';
              second.motionTransfer = { enabled: false, mode, strength: 1 };
              second.transitionLoraEnabled = false;
              second.guideStrength = 0.85;
              ltxDirectorState.duration = 5.5;
              ltxDirectorState.width = 512;
              ltxDirectorState.height = 512;
              ltxDirectorState.fps = 24;
              ltxDirectorState.selectedType = 'segment';
              ltxDirectorState.selectedId = first.id;
              ltxDirectorPendingMotionSegmentId = first.id;
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }""",
                mode,
            )
            page.locator("#ltxDirectorMotionVideoInput").set_input_files(str(guide_path))
            page.wait_for_function(
                """() => {
              const first = (ltxDirectorState.segments || [])[0];
              return !!first?.motionTransfer?.videoSrc && first.motionTransfer.enabled === true;
            }""",
                timeout=120000,
            )
            page.evaluate(
                """() => {
              const [first, second] = ltxDirectorState.segments;
              second.start = Math.max(2.25, Number(first.start || 0) + Number(first.length || 0));
              second.length = Math.max(3.0, Number(ltxDirectorState.duration || 5.5) - Number(second.start || 0));
              ltxDirectorState.duration = Math.max(5.25, Number(second.start || 0) + Number(second.length || 0));
              ltxDirectorConstrainSegment(first, { syncLoadVideo: true });
              ltxDirectorConstrainSegment(second, { syncLoadVideo: true });
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }"""
            )
            payload = page.evaluate("() => collectGenerationPayload()")
            return validate_director_payload(payload, case_name, expect_transition=False, expected_motion_mode=mode)

        def configure_motion_transition_case() -> dict:
            reset_director()
            page.locator("#ltxDirectorImageInput").set_input_files([str(TARGET_IMAGE), str(START_REFERENCE)])
            page.wait_for_function("() => (ltxDirectorState.segments || []).length >= 2", timeout=60000)
            page.evaluate(
                """() => {
              const [first, second] = ltxDirectorState.segments;
              first.start = 0;
              first.length = 2;
              first.prompt = 'Smoke character guided by reference motion, clean cinematic motion';
              first.negativePrompt = 'noise, artifacts, smear, identity drift';
              first.motionTransfer = { enabled: true, mode: 'pose', strength: 1 };
              first.transitionLoraEnabled = false;
              second.start = 2.25;
              second.length = 3.25;
              second.prompt = 'Start frame reference with transition LoRA active toward the selected end frame, preserve identity and continuous motion';
              second.negativePrompt = 'noise, artifacts, smear, identity drift';
              second.motionTransfer = { enabled: false, mode: 'pose', strength: 1 };
              second.transitionLoraEnabled = true;
              second.guideStrength = 1;
              ltxDirectorState.duration = 5.5;
              ltxDirectorState.width = 512;
              ltxDirectorState.height = 512;
              ltxDirectorState.fps = 24;
              ltxDirectorState.selectedType = 'segment';
              ltxDirectorState.selectedId = first.id;
              ltxDirectorPendingMotionSegmentId = first.id;
              ltxDirectorPendingTransitionSegmentId = second.id;
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }"""
            )
            page.locator("#ltxDirectorMotionVideoInput").set_input_files(str(MOTION_GUIDE))
            page.locator("#ltxDirectorTransitionImageInput").set_input_files(str(END_REFERENCE))
            page.wait_for_function(
                """() => {
              const first = (ltxDirectorState.segments || [])[0];
              const second = (ltxDirectorState.segments || [])[1];
              return !!first?.motionTransfer?.videoSrc && first.motionTransfer.enabled === true && second?.transitionLoraEnabled === true && !!second.transitionImageB64;
            }""",
                timeout=120000,
            )
            page.evaluate(
                """() => {
              const [first, second] = ltxDirectorState.segments;
              second.start = Math.max(2.25, Number(first.start || 0) + Number(first.length || 0));
              second.length = Math.max(3.0, Number(ltxDirectorState.duration || 5.5) - Number(second.start || 0));
              ltxDirectorState.duration = Math.max(5.25, Number(second.start || 0) + Number(second.length || 0));
              ltxDirectorConstrainSegment(first, { syncLoadVideo: true });
              ltxDirectorConstrainSegment(second, { syncLoadVideo: true });
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }"""
            )
            payload = page.evaluate("() => collectGenerationPayload()")
            return validate_director_payload(payload, "motion_transition", expect_transition=True, expected_motion_mode="pose")

        def configure_camera_transition_case() -> dict:
            reset_director()
            page.locator("#ltxDirectorImageInput").set_input_files([str(TARGET_IMAGE), str(START_REFERENCE)])
            page.wait_for_function("() => (ltxDirectorState.segments || []).length >= 2", timeout=60000)
            page.evaluate(
                """() => {
              const [first, second] = ltxDirectorState.segments;
              first.start = 0;
              first.length = 2;
              first.prompt = 'same target image subject, same face, same hair, same polka dot outfit, camera orbit from reference video, transition to end frame without identity drift';
              first.negativePrompt = 'different person, changed face, changed gender, identity drift, ghost face, noise, artifacts, smear';
              first.motionTransfer = { enabled: true, mode: 'camera', strength: 1 };
              first.transitionLoraEnabled = true;
              first.guideStrength = 1;
              second.start = 2.25;
              second.length = 3.25;
              second.prompt = 'Plain reference segment without motion transfer, stable clear identity';
              second.negativePrompt = 'noise, artifacts, smear';
              second.motionTransfer = { enabled: false, mode: 'camera', strength: 1 };
              second.transitionLoraEnabled = false;
              ltxDirectorState.duration = 5.5;
              ltxDirectorState.width = 512;
              ltxDirectorState.height = 512;
              ltxDirectorState.fps = 24;
              ltxDirectorState.selectedType = 'segment';
              ltxDirectorState.selectedId = first.id;
              ltxDirectorPendingMotionSegmentId = first.id;
              ltxDirectorPendingTransitionSegmentId = first.id;
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }"""
            )
            page.locator("#ltxDirectorMotionVideoInput").set_input_files(str(CAMERA_MOTION_GUIDE))
            page.locator("#ltxDirectorTransitionImageInput").set_input_files(str(CAMERA_END_FRAME))
            page.wait_for_function(
                """() => {
              const first = (ltxDirectorState.segments || [])[0];
              return !!first?.motionTransfer?.videoSrc && first.motionTransfer.enabled === true && first.transitionLoraEnabled === true && !!first.transitionImageB64;
            }""",
                timeout=120000,
            )
            page.evaluate(
                """() => {
              const [first, second] = ltxDirectorState.segments;
              ltxDirectorConstrainSegment(first, { syncLoadVideo: true });
              second.start = Math.max(2.25, Number(first.start || 0) + Number(first.length || 0));
              second.length = Math.max(3.0, Number(ltxDirectorState.duration || 5.5) - Number(second.start || 0));
              renderLtxDirector();
              syncLtxDirectorWorkflowGraph(false);
            }"""
            )
            payload = page.evaluate("() => collectGenerationPayload()")
            return validate_director_payload(payload, "motion_camera_transition", expect_transition=True, expected_motion_mode="camera")

        def validate_director_payload(payload: dict, case_name: str, expect_transition: bool, expected_motion_mode: str) -> dict:
            director = payload.get("director") or {}
            timeline = director.get("timeline_data") or {}
            motion = timeline.get("motionTransfer") or []
            segments = timeline.get("segments") or []
            if payload.get("workspace") != "director" or payload.get("template") != "LTX_DIRECTOR_SUITE":
                raise AssertionError(f"{case_name}: Director payload did not route through legacy Director: {payload!r}")
            if payload.get("width") != 512 or payload.get("height") != 512 or payload.get("steps") != 4 or payload.get("cfg") != 1:
                raise AssertionError(f"{case_name}: Director payload did not keep 512x512 4 steps cfg 1: {payload!r}")
            if payload.get("video", {}).get("fps") != 24 or payload.get("video", {}).get("motion_transfer_enabled") is not True:
                raise AssertionError(f"{case_name}: Director motion transfer timing did not sync: {payload.get('video')!r}")
            if payload.get("video", {}).get("director_segment_render") is not True or payload.get("workflow_override"):
                raise AssertionError(f"{case_name}: Director Motion/Transition must route through backend segment render, not monolithic workflow_override: {payload.get('video')!r}")
            latent_name = str(payload.get("video", {}).get("latent_upscale") or "")
            if "x2" not in latent_name.lower() or payload.get("video", {}).get("latent_upscale_refine") is not True:
                raise AssertionError(f"{case_name}: Director motion transfer must keep latent upscale/refine compatible like linear LTX: {payload.get('video')!r}")
            distilled = payload.get("distilled_loras") or []
            if len(distilled) != 1 or "384" not in str(distilled[0].get("name") if isinstance(distilled[0], dict) else ""):
                raise AssertionError(f"{case_name}: Director motion transfer must use only the official 384 distilled LoRA stack: {distilled!r}")
            if len(segments) < 2 or not segments[0].get("motionTransfer", {}).get("enabled") or segments[1].get("motionTransfer", {}).get("enabled"):
                raise AssertionError(f"{case_name}: Expected one motion segment and one non-motion segment: {segments!r}")
            first_source_duration = float(segments[0].get("motionTransfer", {}).get("sourceDuration") or 0)
            first_length = float(segments[0].get("length") or 0) / 24.0
            if first_source_duration > 0 and abs(first_length - first_source_duration) > 0.20:
                raise AssertionError(f"{case_name}: Motion segment is not visually/timeline-synced to reference video duration: {segments[0]!r}")
            transition_segments = [segment for segment in segments if segment.get("transitionLoraEnabled")]
            if bool(transition_segments) is not bool(expect_transition):
                raise AssertionError(f"{case_name}: transition state is wrong: {segments!r}")
            if expect_transition and not any(str(segment.get("transitionImageB64") or "").startswith("data:image/") for segment in transition_segments):
                raise AssertionError(f"{case_name}: Transition LoRA segment is missing explicit end frame/reference image: {transition_segments!r}")
            if len(motion) != 1 or motion[0].get("mode") != expected_motion_mode or not str(motion[0].get("videoB64") or "").startswith("data:video/"):
                raise AssertionError(f"{case_name}: Director motion transfer metadata was not serialized: {motion!r}")
            return payload

        def run_director_case(case_name: str) -> dict:
            if case_name == "motion_transition":
                payload = configure_motion_transition_case()
            elif case_name == "motion_camera_transition":
                payload = configure_camera_transition_case()
            elif case_name == "motion_camera_plain":
                payload = configure_motion_plain_case("camera", CAMERA_MOTION_GUIDE, case_name)
            else:
                payload = configure_motion_plain_case()
            job = page.evaluate(
                """async () => {
              const payload = collectGenerationPayload();
              const job = await startGenerationJob(payload);
              return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
            }"""
            )
            outputs = [item for item in (job.get("outputs") or []) if str(item.get("path") or item.get("filename") or "").lower().endswith(".mp4")]
            if not outputs:
                raise AssertionError(f"{case_name}: Director motion transfer job completed without an mp4: {job!r}")
            path = local_output_path(outputs[0])
            metrics = analyze_video(path, f"ltx_director_{case_name}", frames=8, require_motion=True)
            stream = metrics.get("stream") or {}
            actual_duration = float(stream.get("duration") or 0)
            actual_frames = int(float(stream.get("nb_frames") or 0))
            if actual_duration <= 5.0 or actual_frames < 120:
                raise AssertionError(f"{case_name}: Director motion transfer output is too short for the required synced multi-segment battery: {stream!r}")
            if path.parent.name != "videos":
                raise AssertionError(f"{case_name}: Director final output must be in output/videos, got {path}")
            metadata_path = path.with_name(path.name + ".nexus.json")
            if not metadata_path.exists():
                raise AssertionError(f"{case_name}: Director final output did not write metadata sidecar: {metadata_path}")
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            director_meta = metadata.get("director") or {}
            segment_outputs = director_meta.get("segment_outputs") or []
            archive = str(director_meta.get("segment_archive") or "")
            if not archive.startswith("director/") or not archive.endswith("/segments") or not segment_outputs:
                raise AssertionError(f"{case_name}: Director segment archive metadata is missing: {director_meta!r}")
            for relative in segment_outputs:
                segment_path = ROOT / "output" / str(relative).replace("\\", "/")
                if not segment_path.exists() or "\\director\\" not in str(segment_path).lower() and "/director/" not in str(segment_path).lower():
                    raise AssertionError(f"{case_name}: archived Director segment is missing or outside output/director: {relative!r}")
            page.screenshot(path=str(RESULTS / f"front_ltx_director_{case_name}_real.png"), full_page=True)
            return {"case": case_name, "payload": payload, "job": job, "output": str(path), "metrics": metrics}

        results = [
            run_director_case("motion_plain"),
            run_director_case("motion_transition"),
            run_director_case("motion_camera_plain"),
            run_director_case("motion_camera_transition"),
        ]
        browser.close()

    result = {"cases": results}
    (RESULTS / "front_ltx_director_motion_transfer_real.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("ok LTX Director motion transfer real batteries: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
