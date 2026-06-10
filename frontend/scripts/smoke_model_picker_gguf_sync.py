import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"


def job_body(job_id: str) -> str:
    return json.dumps(
        {
            "job_id": job_id,
            "prompt_id": None,
            "status": "completed",
            "progress": 100,
            "message": "model picker gguf sync contract",
            "outputs": [],
            "error": None,
        }
    )


def select_by_hint(page, selector: str, hint: str) -> str:
    value = page.evaluate(
        """({ selector, hint }) => {
          const select = document.querySelector(selector);
          if (!select) throw new Error(`missing ${selector}`);
          const re = new RegExp(hint, 'i');
          const option = [...select.options].find(item => re.test(`${item.value} ${item.textContent} ${item.dataset?.model || ''}`));
          if (!option) return '';
          select.value = option.value;
          select.dispatchEvent(new Event('change', { bubbles: true }));
          updateWorkflowPreview();
          return option.dataset?.model || option.value;
        }""",
        {"selector": selector, "hint": hint},
    )
    return str(value or "")


def active_graph_classes(page) -> list[str]:
    return page.evaluate("() => (activeWorkflowGraph?.nodes || []).map(node => String(node.class_type || ''))")


def prepare_preset(page, preset: str) -> None:
    page.locator(f"button[data-preset='{preset}']").click()
    page.wait_for_function(f"() => activePreset === '{preset}'", timeout=60000)
    page.evaluate(
        """() => {
          document.querySelector('#posPrompt').value = 'front contract test';
          document.querySelector('#widthInput').value = '512';
          document.querySelector('#heightInput').value = '512';
          document.querySelector('#stepsValue').value = '4';
          document.querySelector('#cfgValue').value = '1';
          syncSlider('width');
          syncSlider('height');
          updateSliderFromNumber('steps');
          updateSliderFromNumber('cfg');
          updateWorkflowPreview();
        }"""
    )


def click_generate(page) -> None:
    page.locator("#globalGenerateButton").click()
    page.wait_for_timeout(250)


def main() -> None:
    payloads: list[dict[str, object]] = []

    def capture(route: Route) -> None:
        payloads.append(json.loads(route.request.post_data or "{}"))
        route.fulfill(status=200, content_type="application/json", body=job_body(f"model-picker-{len(payloads)}"))

    def handle_generation_route(route: Route) -> None:
        if route.request.url.rstrip("/").endswith("/api/generate/start"):
            capture(route)
            return
        if route.request.method == "GET":
            route.fulfill(status=200, content_type="application/json", body=job_body("model-picker-status"))
            return
        route.continue_()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        context.add_init_script("localStorage.clear();")
        page = context.new_page()
        page.route("**/api/generate/**", handle_generation_route)
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => typeof collectGenerationPayload === 'function' && backendOnline === true", timeout=120000)

        cases = [
            ("Flux", r"flux.*\.gguf", r"qwen_3_4b|t5xxl|qwen3vl", r"flux2-vae|flux_ae|wan_2\.1_vae"),
            ("LTX", r"ltx.*\.gguf", r"gemma|qwen3vl|t5xxl", r"LTX23_video_vae|wan_2\.1_vae"),
            ("Wan", r"wan.*\.gguf", r"umt5|t5xxl|qwen3vl", r"wan_2\.1_vae|LTX23_video_vae"),
        ]
        selected: dict[str, dict[str, str]] = {}
        for preset, model_hint, encoder_hint, vae_hint in cases:
            prepare_preset(page, preset)
            model = select_by_hint(page, "#modelSelect", model_hint)
            if not model:
                raise AssertionError(f"{preset}: no GGUF model option visible in #modelSelect")
            encoder = select_by_hint(page, "#textEncoderSelect", encoder_hint)
            vae = select_by_hint(page, "#vaeSelect", vae_hint)
            selected[preset] = {"model": model, "encoder": encoder, "vae": vae, "classes": ",".join(active_graph_classes(page))}
            payload = page.evaluate("() => collectGenerationPayload()")
            if str(payload.get("model_name", "")).lower().endswith(".gguf") is False:
                raise AssertionError(f"{preset}: payload did not preserve GGUF model: {payload!r}")
            if encoder and payload.get("text_encoder") != encoder:
                raise AssertionError(f"{preset}: text encoder selection blocked/replaced: selected={encoder!r} payload={payload!r}")
            if vae and payload.get("vae") != vae:
                raise AssertionError(f"{preset}: VAE selection blocked/replaced: selected={vae!r} payload={payload!r}")
            if preset == "Flux" and "UnetLoaderGGUF" not in active_graph_classes(page):
                raise AssertionError(f"{preset}: visual graph did not switch model loader to UnetLoaderGGUF")
            click_generate(page)

        page.screenshot(path=str(RESULTS / "model-picker-gguf-sync.png"), full_page=True)
        browser.close()

    if len(payloads) != 3:
        raise AssertionError(f"Expected 3 generate payloads, got {len(payloads)}")
    (RESULTS / "model-picker-gguf-sync.json").write_text(
        json.dumps({"selected": selected, "payloads": payloads}, indent=2),
        encoding="utf-8",
    )
    print("ok model picker gguf sync: Flux, LTX and Wan preserve model/text_encoder/VAE from /ui")


if __name__ == "__main__":
    main()
