import json
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from visual_checks import RESULTS, analyze_image, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
SAMPLE = ROOT / "input" / "nexus_smoke_reference.png"


CASES = [
    {
        "preset": "Qwen",
        "model_hint": "qwen",
        "control_hint": "qwen_image_canny_diffsynth_controlnet",
        "prompt": "a clean studio portrait, controlled canny composition, detailed but natural",
    },
    {
        "preset": "ZImageTurbo",
        "model_hint": "zimage",
        "control_hint": "Z-Image-Turbo-Fun-Controlnet-Union",
        "prompt": "a clean studio portrait, controlled composition, natural light",
    },
    {
        "preset": "Flux",
        "model_hint": "flux",
        "control_hint": "FLUX.1-dev-ControlNet-Union-Pro-2.0",
        "prompt": "a clean studio portrait, controlled composition, natural light",
    },
]


def select_option_by_hint(page: Page, selector: str, hint: str, required: bool = True) -> str:
    value = page.evaluate(
        """([selector, hint]) => {
          const select = document.querySelector(selector);
          if (!select) return '';
          const needle = String(hint || '').toLowerCase();
          const options = [...select.options];
          const match = options.find(option => [option.value, option.textContent, option.dataset.model]
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


def run_case(page: Page, case: dict[str, str]) -> dict[str, object]:
    preset = case["preset"]
    page.evaluate(
        """([preset, prompt]) => {
          setPreset(document.querySelector(`[data-preset="${preset}"]`), preset);
          switchActivity('txt2img', document.querySelector('[data-activity="txt2img"]'));
          document.querySelector('#posPrompt').value = prompt;
          document.querySelector('#negPrompt').value = 'noise, artifacts, broken anatomy, oversaturated colors';
          document.querySelector('#widthInput').value = '512';
          document.querySelector('#heightInput').value = '512';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          document.querySelector('#controlNetTypeSelect').value = 'canny';
          document.querySelector('#controlNetEnabledToggle').checked = true;
          document.querySelector('#controlNetStrengthSlider').value = '0.85';
          document.querySelector('#controlNetStartInput').value = '0';
          document.querySelector('#controlNetEndInput').value = '1';
          updateControlNetUi();
        }""",
        [preset, case["prompt"]],
    )
    select_option_by_hint(page, "#modelSelect", case["model_hint"], required=False)
    select_option_by_hint(page, "#controlNetModelSelect", case["control_hint"])
    page.locator("#controlNetImageInput").set_input_files(str(SAMPLE))
    page.wait_for_function("() => !!controlNetImageDataUrl && controlNetImageDataUrl.startsWith('data:image/')", timeout=60000)
    payload = page.evaluate("() => collectGenerationPayload()")
    if payload["preset"] != preset:
        raise AssertionError(f"{preset}: payload preset mismatch: {payload['preset']!r}")
    if not payload["controlnet"].get("enabled"):
        raise AssertionError(f"{preset}: ControlNet was not enabled in payload: {payload['controlnet']!r}")
    if not str(payload["controlnet"].get("image") or "").startswith("data:image/"):
        raise AssertionError(f"{preset}: ControlNet image missing from payload")
    if payload["width"] != 512 or payload["height"] != 512 or payload["steps"] != 4:
        raise AssertionError(f"{preset}: wrong battery settings: {payload!r}")

    job = page.evaluate(
        """async () => {
          const payload = collectGenerationPayload();
          const job = await startGenerationJob(payload);
          return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
        }"""
    )
    outputs = job.get("outputs") or []
    if not outputs:
        raise AssertionError(f"{preset}: job completed without outputs: {job!r}")
    path = local_output_path(outputs[0])
    metrics = analyze_image(path, f"controlnet_{preset.lower()}")
    page.screenshot(path=str(RESULTS / f"front_controlnet_{preset.lower()}_real.png"), full_page=True)
    return {"case": case, "payload": payload, "job": job, "output": str(path), "metrics": metrics}


def main() -> None:
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)
        for case in CASES:
            results.append(run_case(page, case))
        browser.close()
    (RESULTS / "front_controlnet_real.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ok front ControlNet real battery: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
