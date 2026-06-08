import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/app"
SAMPLE = ROOT / "temp" / "ideogram4_contract" / "reference.png"
SAMPLE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63600000020001544a0d0a0000000049454e44ae426082"
)


def ensure_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SAMPLE_PNG)


def main() -> None:
    ensure_sample(SAMPLE)
    captured: dict[str, object] = {}

    def capture_generate(route: Route) -> None:
        captured["payload"] = json.loads(route.request.post_data or "{}")
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "job_id": "ideogram4-contract",
                    "prompt_id": None,
                    "status": "completed",
                    "progress": 100,
                    "message": "ideogram4 contract smoke",
                    "outputs": [],
                    "error": None,
                    "created_at": "2026-06-08T00:00:00",
                    "updated_at": "2026-06-08T00:00:00",
                }
            ),
        )

    def fulfill_status(route: Route) -> None:
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps(
                {
                    "template": "Ideogram4",
                    "label": "Ideogram 4",
                    "installed": False,
                    "generation_ready": False,
                    "dependencies_installed": False,
                    "assets": [
                        {"key": "checkpoint", "label": "Ideogram 4 FP8 model", "filename": "ideogram4_fp8_scaled.safetensors", "installed": False, "scope": "model", "size_bytes_min": 9280741285},
                        {"key": "unconditional_checkpoint", "label": "Ideogram 4 unconditional FP8 model", "filename": "ideogram4_unconditional_fp8_scaled.safetensors", "installed": False, "scope": "model", "size_bytes_min": 9280741293},
                        {"key": "qwen3vl", "label": "Ideogram 4 Qwen3-VL text encoder", "filename": "qwen3vl_8b_fp8_scaled.safetensors", "installed": False, "scope": "text_encoder", "size_bytes_min": 10588637512},
                        {"key": "vae", "label": "Flux2 VAE for Ideogram 4", "filename": "flux2-vae.safetensors", "installed": True, "scope": "vae", "size_bytes_min": 336213556},
                        {"key": "gemma4", "label": "Gemma 4 prompt helper encoder", "filename": "gemma4_e4b_it_fp8_scaled.safetensors", "installed": False, "scope": "optional_prompt_helper", "size_bytes_min": 9057782194},
                    ],
                    "missing_assets": [],
                    "missing_required_assets": [
                        {"key": "checkpoint"},
                        {"key": "unconditional_checkpoint"},
                        {"key": "qwen3vl"},
                    ],
                    "missing_optional_assets": [{"key": "gemma4"}],
                    "estimated_missing_required_bytes": 29149754100,
                    "estimated_missing_optional_bytes": 9057782194,
                }
            ),
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.route("**/api/ideogram4/assets/status", fulfill_status)
        page.route("**/api/generate/start", capture_generate)
        page.route(
            "**/api/generate/ideogram4-contract",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "job_id": "ideogram4-contract",
                        "prompt_id": None,
                        "status": "completed",
                        "progress": 100,
                        "message": "ideogram4 contract smoke",
                        "outputs": [],
                        "error": None,
                        "created_at": "2026-06-08T00:00:00",
                        "updated_at": "2026-06-08T00:00:00",
                    }
                ),
            ),
        )
        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.evaluate(
            """
            () => {
              const raw = localStorage.getItem('nexus-generation-state');
              const parsed = raw ? JSON.parse(raw) : { state: {}, version: 0 };
              parsed.state = {
                ...parsed.state,
                preset: 'Ideogram4',
                activity: 'txt2img',
                width: 512,
                height: 512,
                steps: 4,
                cfg: 1,
                sampler: 'euler',
                scheduler: 'simple',
                modelPath: '',
                modelName: '',
                workflowId: '',
                workflowName: '',
                promptRegions: []
              };
              localStorage.setItem('nexus-generation-state', JSON.stringify(parsed));
            }
            """
        )
        page.reload(wait_until="networkidle", timeout=60000)
        page.get_by_placeholder("Describe the image...").fill("Machu Picchu landscape with localized additions")
        page.get_by_role("button", name="ADD obj").click()
        page.locator(".ideogram-region-editor textarea").fill("A white llama standing on the lower-left grassy area")
        page.get_by_role("button", name="ADD obj").click()
        page.locator(".ideogram-region-box").nth(1).click()
        page.locator(".ideogram-region-editor textarea").fill("An old beige desktop PC sitting on the right stone wall")
        page.locator(".ideogram-region-editor input").nth(1).fill("48")
        page.locator(".ideogram-region-editor input").nth(2).fill("54")
        page.locator(".ideogram-region-editor input").nth(3).fill("36")
        page.locator(".ideogram-region-editor input").nth(4).fill("20")
        page.get_by_role("button", name="img2img", exact=True).click()
        page.locator(".img2img-source input[type='file']").first.set_input_files(str(SAMPLE))
        page.get_by_role("button", name="Generate").first.click()
        page.wait_for_timeout(500)
        page.screenshot(path=str(RESULTS / "ideogram4-region-contract.png"), full_page=True)
        browser.close()

    payload = captured.get("payload")
    if not isinstance(payload, dict):
        raise AssertionError("Generate payload was not captured.")
    if payload.get("preset") != "Ideogram4":
        raise AssertionError(f"Expected Ideogram4 preset, got {payload.get('preset')!r}.")
    if payload.get("activity") != "img2img":
        raise AssertionError(f"Expected img2img activity, got {payload.get('activity')!r}.")
    if payload.get("steps") != 4 or float(payload.get("cfg") or 0) != 1:
        raise AssertionError(f"Expected steps=4/cfg=1, got steps={payload.get('steps')!r} cfg={payload.get('cfg')!r}.")
    video = payload.get("video")
    if not isinstance(video, dict) or video.get("ideogram_reference_mode") != "layout_reference_only":
        raise AssertionError(f"Expected Ideogram layout reference flag, got {video!r}.")
    regions = video.get("ideogram_regions")
    if not isinstance(regions, list) or len(regions) < 2:
        raise AssertionError(f"Expected at least two Ideogram regions, got {regions!r}.")
    prompts = " ".join(str(region.get("prompt") or "") for region in regions if isinstance(region, dict)).lower()
    if "llama" not in prompts or "desktop pc" not in prompts:
        raise AssertionError(f"Expected llama and old PC regional prompts, got {regions!r}.")
    print("ok ideogram4 contract: regional boxes, img2img layout guide, 4 steps, cfg 1")


if __name__ == "__main__":
    main()
