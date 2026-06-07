import json
import subprocess
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Page, sync_playwright

from smoke_ltx_motion_transfer_contract import (
    BASE,
    CAMERA_MOTION_GUIDE,
    ROOT,
    RESULTS,
    MOTION_GUIDE,
    TARGET_IMAGE,
    END_REFERENCE,
    backend_fetch,
    click_global_generate,
    collect_payload,
    final_video_output,
    local_output_path,
    poll_job,
)


WIDTH = 512
HEIGHT = 512
STEPS = 4
CFG = 1
FPS = 24
SECONDS = 2


def log(message: str) -> None:
    print(message, flush=True)


def wait_front_backend_sync(page: Page) -> dict[str, object]:
    page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
    page.wait_for_function("() => typeof collectGenerationPayload === 'function' && typeof nexusFetch === 'function'", timeout=120000)
    page.wait_for_function("() => backendOnline === true", timeout=120000)
    return backend_fetch(page, "/health")


def set_value(page: Page, selector: str, value: str) -> None:
    page.locator(selector).fill(value)


def select_if_available(page: Page, selector: str, wanted: str) -> None:
    page.evaluate(
        """({ selector, wanted }) => {
          const el = document.querySelector(selector);
          if (!el) return;
          const option = [...el.options].find(item => item.value === wanted || item.textContent.trim() === wanted);
          if (option) el.value = option.value;
        }""",
        {"selector": selector, "wanted": wanted},
    )


def configure_ltx_base(page: Page, *, condsafe: bool, detailer: bool) -> None:
    page.locator("button[data-preset='LTX']").click()
    page.locator("[data-activity='img2img']").click()
    page.wait_for_function("() => activePreset === 'LTX' && currentActivity === 'img2img'", timeout=60000)
    page.locator("#tab-viewer").click()
    page.wait_for_function("() => activeWorkspace === 'viewer'", timeout=30000)
    page.evaluate(
        """() => {
          clearReferenceImage({ quiet: true });
          if (typeof syncLtxMotionTransferToggle === 'function') syncLtxMotionTransferToggle(false);
          if (document.querySelector('#img2imgModeSelect')) document.querySelector('#img2imgModeSelect').value = 'Image to Image';
          activeWorkflowId = null;
          activeWorkflowGraph = null;
          activeWorkflowAnalysis = null;
        }"""
    )
    set_value(page, "#widthInput", str(WIDTH))
    set_value(page, "#heightInput", str(HEIGHT))
    set_value(page, "#stepsValue", str(STEPS))
    set_value(page, "#cfgValue", str(CFG))
    set_value(page, "#secondsInput", str(SECONDS))
    set_value(page, "#fpsInput", str(FPS))
    page.evaluate(
        """({ condsafe, detailer }) => {
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          const denoise = document.querySelector('#denoiseSlider');
          if (denoise) denoise.value = '1';
          const denoiseValue = document.querySelector('#denoiseValue');
          if (denoiseValue) denoiseValue.value = '1';
          const sampler = document.querySelector('#samplingMethodSelect');
          if (sampler) sampler.value = 'Euler CFG++';
          const scheduler = document.querySelector('#schedulerSelect');
          if (scheduler) scheduler.value = 'Quadratic';
          const latent = document.querySelector('#latentUpscaleSelect');
          if (latent) latent.value = 'ltx-2.3-spatial-upscaler-x2-1.1.safetensors';
          const refine = document.querySelector('#ltxLatentUpscaleRefineToggle');
          if (refine) refine.checked = true;
          const d1 = document.querySelector('#distilledLoraOneSelect');
          const d2 = document.querySelector('#distilledLoraTwoSelect');
          if (d1) d1.value = condsafe ? 'ltx\\\\ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors' : 'None';
          if (d2) d2.value = 'ltx\\\\ltx-2.3-22b-distilled-lora-384-1.1.safetensors';
          const s1 = document.querySelector('#distilledLoraOneStrength');
          const s2 = document.querySelector('#distilledLoraTwoStrength');
          if (s1) s1.value = '0.8';
          if (s2) s2.value = '0.5';
          const detailToggle = document.querySelector('#ltxDetailerToggle');
          if (detailToggle) detailToggle.checked = !!detailer;
          const detailSelect = document.querySelector('#ltxDetailerLoraSelect');
          if (detailer && detailSelect) {
            const concrete = [...detailSelect.options].find(option => /detailer/i.test(option.value + option.textContent) && !/^(none|automatic|auto|off)$/i.test(option.value));
            if (concrete) detailSelect.value = concrete.value;
          }
          syncGenerationActionUi();
          updateWorkflowPreview();
        }""",
        {"condsafe": condsafe, "detailer": detailer},
    )


def set_ltx_start_only(page: Page, *, condsafe: bool, detailer: bool) -> dict[str, object]:
    configure_ltx_base(page, condsafe=condsafe, detailer=detailer)
    page.locator("#referenceImageInput").set_input_files(str(TARGET_IMAGE))
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'LTX'
            && payload?.activity === 'img2img'
            && payload?.workspace === 'viewer'
            && payload?.img2img?.reference_images?.length === 1
            && !payload?.img2img?.base_video
            && payload?.workflow_id == null
            && payload?.workflow_override == null
            && payload?.video?.outpaint_enabled === false
            && payload?.video?.motion_transfer_enabled === false;
        }""",
        timeout=60000,
    )
    return collect_payload(page)


def set_ltx_start_end(page: Page, *, condsafe: bool, detailer: bool) -> dict[str, object]:
    configure_ltx_base(page, condsafe=condsafe, detailer=detailer)
    page.locator("#referenceImageInput").set_input_files([str(TARGET_IMAGE), str(END_REFERENCE)])
    page.wait_for_function(
        """() => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'LTX'
            && payload?.activity === 'img2img'
            && payload?.workspace === 'viewer'
            && payload?.img2img?.reference_images?.length === 2
            && !payload?.img2img?.base_video
            && payload?.workflow_id == null
            && payload?.workflow_override == null
            && payload?.video?.outpaint_enabled === false
            && payload?.video?.motion_transfer_enabled === false
            && payload?.video?.transition_lora_enabled === true;
        }""",
        timeout=60000,
    )
    return collect_payload(page)


def set_ltx_motion_transfer(page: Page, *, mode: str, condsafe: bool, detailer: bool) -> dict[str, object]:
    configure_ltx_base(page, condsafe=condsafe, detailer=detailer)
    guide = CAMERA_MOTION_GUIDE if mode == "camera" else MOTION_GUIDE
    page.locator("#referenceImageInput").set_input_files([str(guide), str(TARGET_IMAGE)])
    page.evaluate("mode => setLtxMotionTransferMode(mode)", mode)
    page.wait_for_function(
        """mode => {
          const payload = collectGenerationPayload();
          return payload?.preset === 'LTX'
            && payload?.activity === 'img2img'
            && payload?.workspace === 'viewer'
            && payload?.img2img?.base_video
            && payload?.img2img?.reference_images?.length >= 1
            && payload?.workflow_id == null
            && payload?.workflow_override == null
            && payload?.video?.outpaint_enabled === false
            && payload?.video?.motion_transfer_enabled === true
            && payload?.video?.motion_transfer_control_mode === mode;
        }""",
        arg=mode,
        timeout=60000,
    )
    return collect_payload(page)


def make_strip(video_path: Path, case: str) -> Path:
    frame_dir = RESULTS / f"frames_ltx_start_matrix_{case}"
    frame_dir.mkdir(exist_ok=True)
    for old in frame_dir.glob("*.png"):
        old.unlink()
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vf",
            "select='eq(n,0)+eq(n,12)+eq(n,24)+eq(n,36)+eq(n,48)',scale=192:192",
            "-vsync",
            "0",
            str(frame_dir / "frame_%02d.png"),
        ],
        cwd=ROOT,
        check=True,
    )
    frames = sorted(frame_dir.glob("frame_*.png"))
    if not frames:
        raise AssertionError(f"{case}: no frames extracted for strip")
    canvas = Image.new("RGB", (192 * len(frames), 224), (10, 10, 12))
    draw = ImageDraw.Draw(canvas)
    for index, frame in enumerate(frames):
        img = Image.open(frame).convert("RGB")
        canvas.paste(img, (index * 192, 0))
        draw.text((index * 192 + 6, 198), frame.name, fill=(230, 230, 230))
    out = RESULTS / f"ltx23-linear-start-matrix-{case}-strip.png"
    canvas.save(out)
    return out


def frame_metrics(strip_path: Path) -> dict[str, float]:
    arr = np.asarray(Image.open(strip_path).convert("RGB"), dtype=np.float32)
    channel_max = arr.max(axis=2)
    channel_min = arr.min(axis=2)
    sat = np.zeros_like(channel_max)
    np.divide(channel_max - channel_min, channel_max, out=sat, where=channel_max > 1)
    local_diff = (np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()) / 2
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "high_sat": float((sat > 0.55).mean()),
        "local_diff": float(local_diff),
    }


def run_case(page: Page, phase: str, case: str, payload: dict[str, object]) -> dict[str, object]:
    log(f"RUN {phase}/{case}")
    job_id = click_global_generate(page)
    job = poll_job(page, job_id, timeout_ms=900000)
    output = final_video_output(job, f"{phase}_{case}")
    path = local_output_path(output)
    strip = make_strip(path, f"{phase}_{case}")
    page.wait_for_function("() => !generationUiActive", timeout=60000)
    return {
        "phase": phase,
        "case": case,
        "payload": {
            "workflow_id": payload.get("workflow_id"),
            "workspace": payload.get("workspace"),
            "steps": payload.get("steps"),
            "cfg": payload.get("cfg"),
            "sampler": payload.get("sampler"),
            "distilled_loras": payload.get("distilled_loras"),
            "img2img": {
                "refs": len((payload.get("img2img") or {}).get("reference_images") or []),
                "base_video": bool((payload.get("img2img") or {}).get("base_video")),
            },
            "video": payload.get("video"),
        },
        "output": str(path),
        "strip": str(strip),
        "metrics": frame_metrics(strip),
    }


def main() -> None:
    missing = [path for path in (TARGET_IMAGE, END_REFERENCE, MOTION_GUIDE, CAMERA_MOTION_GUIDE) if not path.exists()]
    if missing:
        raise FileNotFoundError(", ".join(str(path) for path in missing))
    variants = [
        ("only384_no_detailer", False, False),
        ("condsafe384_no_detailer", True, False),
        ("only384_detailer", False, True),
        ("condsafe384_detailer", True, True),
    ]
    summary = {"started_at": time.strftime("%Y-%m-%d %H:%M:%S"), "cases": []}
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.set_default_timeout(3600000)
        page.on("dialog", lambda dialog: dialog.accept())
        health = wait_front_backend_sync(page)
        summary["health"] = health
        log(f"FRONT READY health={health}")
        for case, condsafe, detailer in variants:
            summary["cases"].append(run_case(page, "start_only", case, set_ltx_start_only(page, condsafe=condsafe, detailer=detailer)))
        for case, condsafe, detailer in variants:
            summary["cases"].append(run_case(page, "start_end", case, set_ltx_start_end(page, condsafe=condsafe, detailer=detailer)))
        for mode in ("pose", "depth", "canny", "camera"):
            summary["cases"].append(run_case(page, "motion_transfer", mode, set_ltx_motion_transfer(page, mode=mode, condsafe=False, detailer=False)))
        browser.close()
    out = RESULTS / "ltx23_linear_start_matrix.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
