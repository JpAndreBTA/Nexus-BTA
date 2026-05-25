from pathlib import Path

from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/app"


def main() -> None:
    routes = [
        ("studio", "/", "studio"),
        ("extras", "/extras", "extras"),
        ("gallery", "/gallery", "gallery"),
        ("workflow", "/workflow", "workflow"),
        ("models", "/models", "models"),
        ("settings", "/settings", "settings"),
    ]
    viewports = [
        ("desktop", {"width": 1440, "height": 900}),
        ("mobile", {"width": 390, "height": 844}),
    ]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        for route_name, path, route_id in routes:
            for viewport_name, viewport in viewports:
                page = browser.new_page(viewport=viewport)
                page.goto(f"{BASE}{path}", wait_until="networkidle", timeout=60000)
                page.locator(f".route-sentinel[data-route='{route_id}']").wait_for(state="attached", timeout=30000)
                if page.locator("h1").count():
                    raise AssertionError(f"{path} should not render visible page titles.")
                page.screenshot(path=str(RESULTS / f"app-smoke-{route_name}-{viewport_name}.png"), full_page=True)
                print(f"ok {route_name} {viewport_name}: {route_id}")
                page.close()
        browser.close()


if __name__ == "__main__":
    main()
