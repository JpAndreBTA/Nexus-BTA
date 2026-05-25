import json
from pathlib import Path

from playwright.sync_api import Route, sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/app"
SAMPLE = ROOT / "temp" / "extras_validation" / "sample.png"
SAMPLE_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d49484452000000010000000108060000001f15c489"
    "0000000a49444154789c63600000020001544a0d0a0000000049454e44ae426082"
)


def ensure_sample(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(SAMPLE_PNG + path.stem.encode("ascii"))


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
                    "job_id": "contract-smoke",
                    "prompt_id": None,
                    "status": "completed",
                    "progress": 100,
                    "message": "contract smoke",
                    "outputs": [],
                    "error": None,
                    "created_at": "2026-05-25T00:00:00",
                    "updated_at": "2026-05-25T00:00:00",
                }
            ),
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.route("**/api/generate/start", capture_generate)
        page.route(
            "**/api/generate/contract-smoke",
            lambda route: route.fulfill(
                status=200,
                content_type="application/json",
                body=json.dumps(
                    {
                        "job_id": "contract-smoke",
                        "prompt_id": None,
                        "status": "completed",
                        "progress": 100,
                        "message": "contract smoke",
                        "outputs": [],
                        "error": None,
                        "created_at": "2026-05-25T00:00:00",
                        "updated_at": "2026-05-25T00:00:00",
                    }
                ),
            ),
        )

        page.goto(f"{BASE}/workflow", wait_until="networkidle", timeout=60000)
        page.locator(".workflow-node-card").first.wait_for(timeout=30000)
        page.get_by_role("button", name="Activate").click()

        page.goto(BASE, wait_until="networkidle", timeout=60000)
        page.get_by_role("button", name="img2img", exact=True).click()
        page.get_by_placeholder("Describe the image...").fill("contract smoke inpaint")
        page.locator(".img2img-source input[type='file']").first.set_input_files(str(SAMPLE))
        page.get_by_role("button", name="Inpaint Canvas").click()

        canvas = page.locator(".studio-inpaint-workspace canvas[aria-label='Inpaint mask canvas']").first
        canvas.wait_for(timeout=30000)
        page.wait_for_function(
            """
            () => {
              const canvas = document.querySelector('.studio-inpaint-workspace canvas[aria-label="Inpaint mask canvas"]');
              return canvas && canvas.width > 16 && canvas.height > 16;
            }
            """
        )
        canvas.evaluate(
            """
            canvas => {
              const rect = canvas.getBoundingClientRect();
              const eventInit = (x, y) => ({
                bubbles: true,
                pointerId: 1,
                pointerType: 'mouse',
                clientX: rect.left + rect.width * x,
                clientY: rect.top + rect.height * y
              });
              canvas.dispatchEvent(new PointerEvent('pointerdown', eventInit(0.25, 0.45)));
              canvas.dispatchEvent(new PointerEvent('pointermove', eventInit(0.45, 0.50)));
              canvas.dispatchEvent(new PointerEvent('pointermove', eventInit(0.75, 0.55)));
              canvas.dispatchEvent(new PointerEvent('pointerup', eventInit(0.75, 0.55)));
            }
            """
        )
        masked = canvas.evaluate(
            """
            canvas => {
              const data = canvas.getContext('2d').getImageData(0, 0, canvas.width, canvas.height).data;
              for (let index = 3; index < data.length; index += 4) if (data[index] > 3) return true;
              return false;
            }
            """
        )
        if not masked:
            raise AssertionError("Inpaint canvas did not record mask pixels.")

        page.locator("details.control-section").filter(has_text="ControlNet / Reference").locator("summary").click()
        page.get_by_label("Toggle ControlNet").click()
        page.locator(".controlnet-panel input[type='file']").set_input_files(str(SAMPLE))
        page.get_by_role("button", name="Generate").first.click()
        page.wait_for_timeout(300)
        page.screenshot(path=str(RESULTS / "app-smoke-generation-contract.png"), full_page=True)
        browser.close()

    payload = captured.get("payload")
    if not isinstance(payload, dict):
        raise AssertionError("Generate payload was not captured.")
    if payload.get("activity") != "img2img":
        raise AssertionError(f"Expected img2img activity, got {payload.get('activity')!r}.")
    if not payload.get("workflow_id"):
        raise AssertionError("Expected an active workflow_id in the payload.")

    img2img = payload.get("img2img")
    if not isinstance(img2img, dict) or "inpaint" not in str(img2img.get("mode", "")).lower():
        raise AssertionError(f"Expected inpaint mode, got {img2img!r}.")
    if not str(img2img.get("mask_image") or "").startswith("data:image/png"):
        raise AssertionError("Expected mask_image to be a PNG data URL.")

    controlnet = payload.get("controlnet")
    if not isinstance(controlnet, dict) or not controlnet.get("enabled"):
        raise AssertionError(f"Expected enabled ControlNet, got {controlnet!r}.")
    if not str(controlnet.get("image") or "").startswith("data:image/"):
        raise AssertionError("Expected ControlNet image data URL.")

    print("ok generation contract: inpaint mask + controlnet + workflow_id")


if __name__ == "__main__":
    main()
