import json
import sys
from pathlib import Path

from playwright.sync_api import Route, sync_playwright

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from backend.nexus_backend.schemas import GenerateRequest
from backend.nexus_backend.workflows import build_basic_ltx_img2video_workflow


RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
START = ROOT / "input" / "Smoke_Character.png"
END = ROOT / "input" / "Smoke_EndFrame.png"


def job_body(job_id: str) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "prompt_id": None,
            "status": "completed",
            "progress": 100,
            "message": "ltx loop cycle contract",
            "outputs": [],
            "error": None,
            "created_at": "2026-06-01T00:00:00",
            "updated_at": "2026-06-01T00:00:00",
        }
    )


def configure_ltx_base(page) -> None:
    page.locator("button[data-preset='LTX']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'LTX' && currentActivity === 'img2img'", timeout=60000)
    page.locator("#tab-viewer").click()
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncLtxLoopCycleToggle === 'function') syncLtxLoopCycleToggle(false);
          if (typeof syncLtxMotionTransferToggle === 'function') syncLtxMotionTransferToggle(false);
          document.querySelector('#widthInput').value = '512';
          document.querySelector('#heightInput').value = '512';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          document.querySelector('#fpsInput').value = '24';
          document.querySelector('#secondsInput').value = '2';
          document.querySelector('#framesInput').value = '49';
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler CFG++';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Quadratic';
          const latent = document.querySelector('#latentUpscaleSelect');
          if (latent) latent.value = 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors';
          const refine = document.querySelector('#ltxLatentUpscaleRefineToggle');
          if (refine) refine.checked = true;
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          syncVideoMotionFields('framesInput');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
    )


def verify_backend_livewallpaper_contract() -> None:
    def build_workflow(available_nodes: set[str]) -> dict[str, object]:
        return build_basic_ltx_img2video_workflow(
            request,
            checkpoint_name="DasiwaLTX23Lightspeed_solsticecoinV2.safetensors",
            text_encoder_name="gemma-3-12b-it-heretic-v2_nvfp4.safetensors",
            reference_image_name="Smoke_splashART.jpeg",
            reference_end_image_name="Smoke_splashART.jpeg",
            video_vae_name="LTX23_video_vae_bf16.safetensors",
            audio_vae_name="LTX23_audio_vae_bf16.safetensors",
            latent_upscale_name="ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
            detailer_lora_name="ltx_ic\\ltx-2-19b-ic-lora-detailer.safetensors",
            available_nodes=available_nodes,
        )

    request = GenerateRequest(
        activity="img2img",
        preset="LTX",
        model_name="DasiwaLTX23Lightspeed_solsticecoinV2.safetensors",
        width=928,
        height=480,
        steps=4,
        cfg=1,
        sampler="Euler CFG++",
        scheduler="Quadratic",
        prompt="cinematic splash art loop",
        negative_prompt="noise",
        loras=[
            {
                "name": "ltx\\livewallpaper_ltx23_r64_6250.safetensors",
                "relative_name": "ltx\\livewallpaper_ltx23_r64_6250.safetensors",
                "strength": 0.35,
                "strength_model": 0.35,
            }
        ],
        video={
            "ltx_loop_cycle": True,
            "ltx_loop_source": "start_frame_as_end_frame",
            "fps": 24,
            "seconds": 5,
            "frames": 121,
            "latent_upscale_refine": True,
            "detailer_enabled": True,
            "detailer_lora": "Automatic",
            "omnicine_enabled": True,
            "omnicine_lora": "ltx\\Singularity LTX-2.3  OmniCine Preview v0.1.safetensors",
            "ltx_loop_mid_motion_guide": True,
            "active_audio": True,
            "audio_vae": "LTX23_audio_vae_bf16.safetensors",
        },
    )
    base_nodes = {"LTX2LoraLoaderAdvanced", "LTXVLoopingSampler", "STGGuiderAdvanced", "LTXVAddGuide"}
    workflow = build_workflow(base_nodes)
    if workflow.get("12", {}).get("class_type") != "SamplerCustomAdvanced":
        raise AssertionError("LTX native looping sampler must stay opt-in to avoid loop-cycle quality regressions")
    prompt_text = " ".join(str(value) for node in workflow.values() for value in node.get("inputs", {}).values())
    if "l1v3w4llp4p3r" not in prompt_text:
        raise AssertionError("livewallpaper LTX concept LoRA did not inject its trigger word")
    lora_nodes = [node for node in workflow.values() if "livewallpaper_ltx23_r64_6250" in str(node.get("inputs", "")).lower()]
    if not lora_nodes or lora_nodes[0].get("class_type") != "LTX2LoraLoaderAdvanced":
        raise AssertionError(f"livewallpaper LTX concept LoRA did not use the advanced LTX loader: {lora_nodes!r}")
    detailer_nodes = [node for node in workflow.values() if "ltx-2-19b-ic-lora-detailer" in str(node.get("inputs", "")).lower()]
    if not detailer_nodes:
        raise AssertionError("LTX loop concept LoRA path lost detailer compatibility")
    detailer_titles = " ".join(str(node.get("_meta", {}).get("title", "")) for node in detailer_nodes)
    if "Refiner Detailer" not in detailer_titles:
        raise AssertionError(f"LTX loop latent-upscale refine did not receive the detailer LoRA: {detailer_nodes!r}")
    omni_nodes = [node for node in workflow.values() if "omnicine" in str(node.get("inputs", "")).lower() or "singularity" in str(node.get("inputs", "")).lower()]
    if not omni_nodes:
        raise AssertionError("LTX loop lost OmniCine LoRA compatibility")
    for node in [*lora_nodes, *omni_nodes]:
        inputs = node.get("inputs", {})
        if float(inputs.get("video_to_audio") or 0) <= 0 or float(inputs.get("audio") or 0) <= 0 or float(inputs.get("audio_to_video") or 0) <= 0:
            raise AssertionError(f"LTX audio-enabled LoRA must route audio strengths: {node!r}")
    required_audio_nodes = {
        "16": "VAELoader",
        "17": "LTXVEmptyLatentAudio",
        "18": "LTXVConcatAVLatent",
        "19": "LTXVSeparateAVLatent",
        "20": "LTXVAudioVAEDecode",
    }
    for node_id, class_type in required_audio_nodes.items():
        if workflow.get(node_id, {}).get("class_type") != class_type:
            raise AssertionError(f"LTX loop audio node {node_id} must be {class_type}: {workflow.get(node_id)!r}")
    if any(node.get("class_type") == "AudioVolumeNormalization" for node in workflow.values()):
        raise AssertionError("LTX loop active_audio must not require AudioVolumeNormalization when the Comfy node is missing")
    save_inputs = workflow.get("15", {}).get("inputs", {})
    create_inputs = workflow.get("14", {}).get("inputs", {})
    if "audio" not in save_inputs and "audio" not in create_inputs:
        raise AssertionError("LTX loop active_audio must attach decoded audio to the final video output")
    if create_inputs.get("audio") != ["20", 0]:
        raise AssertionError(f"LTX loop active_audio must connect decoded audio directly when normalization is missing: {create_inputs.get('audio')!r}")
    normalized_workflow = build_workflow(base_nodes | {"AudioVolumeNormalization"})
    if normalized_workflow.get("33", {}).get("class_type") != "AudioVolumeNormalization":
        raise AssertionError("LTX loop active_audio should use AudioVolumeNormalization when the Comfy node is available")
    if normalized_workflow.get("14", {}).get("inputs", {}).get("audio") != ["33", 0]:
        raise AssertionError("LTX loop active_audio should route final video audio through normalization when available")
    mid_nodes = [
        node
        for node in workflow.values()
        if "mid motion" in str(node.get("_meta", {}).get("title", "")).lower()
        or "loop mid motion" in str(node.get("_meta", {}).get("title", "")).lower()
    ]
    if mid_nodes:
        raise AssertionError(f"LTX loop must not inject the removed mid-motion guide: {mid_nodes!r}")
    if workflow.get("76", {}).get("class_type") != "LTXVAddGuide":
        raise AssertionError("LTX loop FLF2V must keep the start endpoint guide")
    if workflow.get("178", {}).get("class_type") != "LTXVAddGuide":
        raise AssertionError("LTX loop FLF2V must keep the final endpoint guide")
    if "81" not in workflow and "540" not in workflow:
        raise AssertionError("LTX loop FLF2V must crop guide conditioning before decode/upscale")


def click_generate(page) -> None:
    page.locator("#globalGenerateButton").click()
    page.wait_for_timeout(250)


def main() -> None:
    for item in (START, END):
        if not item.exists():
            raise AssertionError(f"Missing smoke input: {item}")
    payloads: list[dict[str, object]] = []

    def capture_generate(route: Route) -> None:
        payloads.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body=job_body(f"ltx-loop-contract-{len(payloads)}"))

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.on("dialog", lambda dialog: dialog.accept())
        page.route("**/api/ltx/**/status", lambda route: route.fulfill(status=200, content_type="application/json", body=json.dumps({"installed": True, "name": "ltx_transition\\ltx2.3-transition.safetensors"})))
        page.route("**/api/generate/start", capture_generate)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => typeof collectGenerationPayload === 'function'", timeout=120000)

        configure_ltx_base(page)
        page.locator("#referenceImageInput").set_input_files(str(START))
        page.wait_for_function("() => collectGenerationPayload()?.video?.ltx_loop_cycle === false", timeout=60000)
        click_generate(page)

        page.evaluate("() => syncLtxLoopCycleToggle(true)")
        page.wait_for_function(
            """() => {
              const payload = collectGenerationPayload();
              return payload?.preset === 'LTX'
                && payload?.activity === 'img2img'
                && payload?.img2img?.reference_images?.length === 2
                && payload?.video?.ltx_loop_cycle === true
                && payload?.video?.ltx_loop_source === 'start_frame_as_end_frame'
                && payload?.video?.transition_lora_enabled === false
                && payload?.video?.motion_strength === 0.30
                && payload?.video?.start_frame_strength === 0.70
                && payload?.video?.end_frame_strength === 0.70
                && payload?.video?.latent_upscale_refine === true
                && payload?.video?.ltx_endpoint_frame_lock === false
                && payload?.video?.ltx_loop_post_seam_blend === false
                && payload?.video?.ltx_loop_mid_motion_guide === false
                && payload?.workflow_id == null
                && payload?.workflow_override == null;
            }""",
            timeout=60000,
        )
        click_generate(page)

        configure_ltx_base(page)
        page.locator("#referenceImageInput").set_input_files([str(START), str(END)])
        page.wait_for_function(
            """() => {
              const payload = collectGenerationPayload();
              return payload?.img2img?.reference_images?.length === 2
                && payload?.video?.ltx_loop_cycle === false
                && payload?.video?.transition_lora_enabled === true;
            }""",
            timeout=60000,
        )
        click_generate(page)
        page.screenshot(path=str(RESULTS / "ltx23-loop-cycle-contract.png"), full_page=True)
        browser.close()

    if len(payloads) != 3:
        raise AssertionError(f"Expected 3 payloads, captured {len(payloads)}")
    normal_start, loop_start, normal_start_end = payloads
    if normal_start["video"].get("ltx_loop_cycle") is not False:
        raise AssertionError(f"normal LTX start was contaminated by loop: {normal_start!r}")
    if loop_start["video"].get("ltx_loop_cycle") is not True:
        raise AssertionError(f"LTX loop toggle did not sync: {loop_start!r}")
    if loop_start["video"].get("transition_lora_enabled") is not False:
        raise AssertionError(f"LTX loop should not enable transition LoRA: {loop_start!r}")
    if loop_start["video"].get("motion_strength") != 0.30:
        raise AssertionError(f"LTX loop should lower start-image conditioning enough for motion: {loop_start!r}")
    if loop_start["video"].get("start_frame_strength") != 0.70 or loop_start["video"].get("end_frame_strength") != 0.70:
        raise AssertionError(f"LTX loop should use balanced FLF2V guide strengths: {loop_start!r}")
    if loop_start["video"].get("latent_upscale_refine") is not True:
        raise AssertionError(f"LTX loop should respect the user's latent upscale refine toggle: {loop_start!r}")
    if loop_start["video"].get("ltx_endpoint_frame_lock") is not False:
        raise AssertionError(f"LTX loop should not hard-lock endpoint frames in post: {loop_start!r}")
    if loop_start["video"].get("ltx_loop_mid_motion_guide") is not False:
        raise AssertionError(f"LTX loop should not use the removed mid-motion guide: {loop_start!r}")
    if normal_start["video"].get("motion_strength") == 0.30:
        raise AssertionError(f"normal LTX start inherited loop-only tuning: {normal_start!r}")
    if normal_start_end["video"].get("ltx_loop_cycle") is not False:
        raise AssertionError(f"normal LTX start+end was contaminated by loop: {normal_start_end!r}")
    if normal_start_end["video"].get("ltx_endpoint_frame_lock") is False:
        raise AssertionError(f"normal LTX start+end inherited loop endpoint-lock override: {normal_start_end!r}")
    verify_backend_livewallpaper_contract()
    print("ok ltx loop cycle contract: loop isolated and synced")


if __name__ == "__main__":
    main()
