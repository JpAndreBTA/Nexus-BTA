import json
import os
from pathlib import Path

from PIL import Image
from playwright.sync_api import Page, sync_playwright

from visual_checks import RESULTS, analyze_image, local_output_path


ROOT = Path(__file__).resolve().parents[2]
BASE = "http://127.0.0.1:7861/ui"
BASE_IMAGE = ROOT / "input" / "SmokeTest.png"
TEST_CASES = {item.strip() for item in os.environ.get("IDEOGRAM4_SMOKE_CASES", "txt2img,img2img").split(",") if item.strip()}
TEST_WIDTH = int(os.environ.get("IDEOGRAM4_SMOKE_WIDTH", "912"))
TEST_HEIGHT = int(os.environ.get("IDEOGRAM4_SMOKE_HEIGHT", "512"))
TEST_STEPS = int(os.environ.get("IDEOGRAM4_SMOKE_STEPS", "12"))
TEST_CFG = float(os.environ.get("IDEOGRAM4_SMOKE_CFG", "1"))

TXT2IMG_JSON_PROMPT = json.dumps(
    {
        "high_level_description": "A single coherent editorial photograph of one chef standing in a bright professional kitchen.",
        "style_description": {
            "medium": "photograph",
            "lighting": "balanced natural window light with soft shadows",
            "aesthetics": "clean, sharp, realistic, uncluttered composition",
            "photo": "high resolution portrait, recognizable face, natural skin and fabric texture",
        },
        "compositional_deconstruction": {
            "background": "modern stainless kitchen counters, warm neutral walls, utensils visible but not cluttered",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [80, 260, 960, 740],
                    "desc": "one adult chef facing camera, white chef hat, dark jacket, hands gently holding a small mixing bowl, calm expression",
                }
            ],
        },
    },
    separators=(",", ":"),
)

IMG2IMG_JSON_PROMPT = json.dumps(
    {
        "high_level_description": "Preserve the reference chef as a single coherent editorial kitchen portrait.",
        "style_description": {
            "medium": "photograph",
            "lighting": "soft natural kitchen light",
            "aesthetics": "clean realistic details, recognizable face, stable clothing and pose",
            "photo": "sharp documentary portrait with intact hands and background",
        },
        "compositional_deconstruction": {
            "background": "same kitchen layout and warm room tone from the reference image",
            "elements": [
                {
                    "type": "obj",
                    "bbox": [60, 240, 980, 760],
                    "desc": "the same chef from the reference image, centered, white chef hat, dark patterned jacket, hands together near the counter",
                }
            ],
        },
    },
    separators=(",", ":"),
)


def select_option_by_hint(page: Page, selector: str, hint: str) -> str:
    return page.evaluate(
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


def setup_ideogram(page: Page, activity: str, prompt: str) -> dict[str, object]:
    page.evaluate(
        """([activity, prompt, width, height, steps, cfg]) => {
          setPreset(document.querySelector('[data-preset="Ideogram4"]'), 'Ideogram4');
          switchActivity(activity, document.querySelector(`[data-activity="${activity}"]`));
          document.querySelector('#tab-viewer')?.click();
          document.querySelector('#posPrompt').value = prompt;
          document.querySelector('#negPrompt').value = 'blur, noise, unreadable face, warped hands, broken layout';
          document.querySelector('#widthInput').value = String(width);
          document.querySelector('#heightInput').value = String(height);
          document.querySelector('#stepsValue').value = String(steps);
          document.querySelector('#cfgValue').value = String(cfg);
          const method2 = document.querySelector('#ideogramMethod2NoiseSamplerSelect');
          if (method2) method2.value = 'pyramid';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          syncGenerationActionUi();
          updateWorkflowPreview();
        }""",
        [activity, prompt, TEST_WIDTH, TEST_HEIGHT, TEST_STEPS, TEST_CFG],
    )
    page.wait_for_function(f"() => activePreset === 'Ideogram4' && currentActivity === '{activity}'", timeout=60000)
    select_option_by_hint(page, "#modelSelect", "ideogram4")
    select_option_by_hint(page, "#vaeSelect", "flux2-vae")
    select_option_by_hint(page, "#textEncoderSelect", "qwen3vl")
    if activity == "img2img":
        page.locator("#referenceImageInput").set_input_files(str(BASE_IMAGE))
        page.wait_for_function("() => !!referenceImageDataUrl && collectGenerationPayload()?.img2img?.reference_images?.length >= 1", timeout=60000)
    return page.evaluate("() => collectGenerationPayload()")


def setup_ideogram_boxes(page: Page) -> dict[str, object]:
    setup_ideogram(page, "img2img", IMG2IMG_JSON_PROMPT)
    page.evaluate(
        """() => {
          ideogramRegions = [
            {
              id: 'smoke_obj_box',
              type: 'obj',
              x: 18,
              y: 18,
              w: 48,
              h: 68,
              prompt: 'preserve the chef as one coherent object, white chef hat, dark jacket, natural hands',
              color: 'white'
            },
            {
              id: 'smoke_text_box',
              type: 'text',
              x: 66,
              y: 12,
              w: 22,
              h: 16,
              prompt: 'small readable label text: NEXUS',
              color: 'black text on warm paper'
            }
          ];
          ideogramRegionActiveId = 'smoke_text_box';
          if (typeof renderIdeogramRegions === 'function') renderIdeogramRegions();
          syncGenerationActionUi();
          updateWorkflowPreview();
        }"""
    )
    page.wait_for_function(
        """() => {
          const regions = collectGenerationPayload()?.video?.ideogram_regions || [];
          return regions.length === 2
            && regions.some(region => region.type === 'obj')
            && regions.some(region => region.type === 'text')
            && regions.every(region => Array.isArray(region.bbox) && region.bbox.length === 4);
        }""",
        timeout=60000,
    )
    return page.evaluate("() => collectGenerationPayload()")


def run_generation(page: Page, case: str) -> dict[str, object]:
    payload = page.evaluate("() => collectGenerationPayload()")
    if payload["width"] != TEST_WIDTH or payload["height"] != TEST_HEIGHT or payload["steps"] != TEST_STEPS or float(payload["cfg"]) != TEST_CFG:
        raise AssertionError(f"{case}: payload did not respect side menu: {payload!r}")
    if payload.get("video", {}).get("ideogram_method2_noise_sampler") != "pyramid":
        raise AssertionError(f"{case}: Ideogram Method 2 sampler missing from frontend payload: {payload!r}")
    if "boxes" in case:
        regions = payload.get("video", {}).get("ideogram_regions") or []
        types = {region.get("type") for region in regions}
        if len(regions) != 2 or types != {"obj", "text"}:
            raise AssertionError(f"{case}: Ideogram obj/text regions not synced: {regions!r}")
    job = page.evaluate(
        """async () => {
          const payload = collectGenerationPayload();
          const job = await startGenerationJob(payload);
          return await pollGenerationJob(job.job_id, payload, { skipGallery: true });
        }"""
    )
    outputs = job.get("outputs") or []
    if not outputs:
        raise AssertionError(f"{case}: generation completed without outputs: {job!r}")
    output_path = local_output_path(outputs[0])
    metrics = analyze_image(output_path, case)
    image = Image.open(output_path)
    if image.size != (TEST_WIDTH, TEST_HEIGHT):
        raise AssertionError(f"{case}: output size {image.size} does not match requested {TEST_WIDTH}x{TEST_HEIGHT}")
    page.screenshot(path=str(RESULTS / f"{case}.png"), full_page=True)
    return {"case": case, "payload": payload, "job": job, "output": str(output_path), "metrics": metrics}


def main() -> None:
    if not BASE_IMAGE.exists():
        raise AssertionError(f"Missing smoke input: {BASE_IMAGE}")
    results = []
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof collectGenerationPayload === 'function'", timeout=120000)

        if "txt2img" in TEST_CASES:
            setup_ideogram(page, "txt2img", TXT2IMG_JSON_PROMPT)
            results.append(run_generation(page, "ideogram4_txt2img_12steps_real"))

        if "img2img" in TEST_CASES:
            setup_ideogram(page, "img2img", IMG2IMG_JSON_PROMPT)
            results.append(run_generation(page, "ideogram4_img2img_12steps_real"))
        if "boxes" in TEST_CASES:
            setup_ideogram_boxes(page)
            results.append(run_generation(page, "ideogram4_img2img_boxes_method2_12steps_real"))
        browser.close()
    (RESULTS / "ideogram4_real_visual.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print("ok ideogram4 real visual: " + ", ".join(item["output"] for item in results))


if __name__ == "__main__":
    main()
