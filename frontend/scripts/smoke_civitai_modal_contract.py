import json

from playwright.sync_api import Route, sync_playwright


BASE = "http://127.0.0.1:7861/index.html"


def fake_items(offset: int = 0) -> list[dict[str, object]]:
    items = [
        {
            "id": 1000 + offset + index,
            "name": f"Contract {'mature ' if index == 1 else ''}model {offset + index}",
            "type": "LORA" if index % 2 else "Checkpoint",
            "creator": "nexus",
            "nsfw": False,
            "installed": index == 0,
            "relative_path": "models/loras/installed.safetensors" if index == 0 else "",
            "tags": ["character", "style", "nsfw" if index == 1 else "photorealistic"],
            "stats": {"downloadCount": 100 + index},
            "versions": [
                {
                    "id": 2000 + offset + index,
                    "name": "v1",
                    "base_model": "SDXL 1.0",
                    "file_name": "contract.safetensors",
                    "file_size_kb": 1024,
                    "installed": index == 0,
                    "relative_path": "models/loras/installed.safetensors" if index == 0 else "",
                    "preview": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.jpeg",
                    "previews": [
                        {
                            "url": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.jpeg",
                            "type": "image",
                            "nsfw": False,
                        },
                        {
                            "url": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.mp4",
                            "type": "video",
                            "poster": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.jpeg",
                            "videoUrl": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.mp4",
                            "nsfw": False,
                        },
                    ],
                    "url": "https://civitai.com/models/1000",
                }
            ],
            "preview": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.jpeg",
            "url": "https://civitai.com/models/1000",
        }
        for index in range(4)
    ]
    items[2]["versions"][0]["preview"] = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.mp4"
    items[2]["versions"][0]["previews"] = [
        {
            "url": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.mp4",
            "type": "video",
            "videoUrl": "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.mp4",
            "nsfw": False,
        }
    ]
    items[2]["preview"] = "https://image.civitai.com/xG1nkqKTMzGDvpLrqFT7WA/demo.mp4"
    items[3]["name"] = f"Splash Art Contract {offset}"
    items[3]["tags"] = ["splash", "art", "style"]
    return items


def main() -> None:
    captured: list[dict[str, object]] = []

    def capture_search(route: Route) -> None:
        payload = json.loads(route.request.post_data or "{}")
        captured.append(payload)
        cursor = "cursor-2" if not payload.get("cursor") else None
        route.fulfill(
            status=200,
            content_type="application/json",
            body=json.dumps({"items": fake_items(len(captured) * 10), "metadata": {"nextCursor": cursor}}),
        )

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 900})
        page.on("console", lambda msg: print(f"browser console: {msg.type}: {msg.text}") if msg.type == "error" else None)
        page.route("**/api/civitai/search", capture_search)
        page.route("**/api/civitai/downloads", lambda route: route.fulfill(status=200, content_type="application/json", body='{"jobs": []}'))
        page.goto(BASE, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_function("() => !!window.NexusCivitai", timeout=30000)

        page.evaluate("window.NexusCivitai.open()")
        page.wait_for_selector("#civitaiSearchGrid article", timeout=30000)
        first_payload = captured[-1]
        if int(first_payload.get("limit") or 0) < 30:
            raise AssertionError(f"Lazy page size too low: {first_payload.get('limit')}")
        page.uncheck("#civitaiBlurMatureToggle")
        page.wait_for_timeout(500)
        if not page.locator("#civitaiSearchGrid video.nexus-civitai-tile-video").count():
            raise AssertionError("MP4 preview without poster did not render as a lazy video tile.")
        page.locator("#civitaiSearchGrid article", has_text="Contract model 12").click()
        page.wait_for_timeout(250)
        if not page.locator("#civitaiResultPanel video.nexus-civitai-tile-video").count():
            raise AssertionError("Detail preview did not render the MP4 model preview.")
        page.get_by_role("button", name="Back to exploration list").click()
        page.wait_for_timeout(250)
        page.uncheck("#civitaiNsfwToggle")
        page.wait_for_timeout(250)
        if page.locator("#civitaiSearchGrid article", has_text="Contract mature").count():
            raise AssertionError("Mature-tagged model stayed visible with Show mature disabled.")
        page.check("#civitaiNsfwToggle")
        page.wait_for_timeout(250)
        page.check("#civitaiInstalledToggle")
        page.wait_for_timeout(250)
        installed_count = page.locator("#civitaiSearchGrid article").count()
        if installed_count != 1:
            raise AssertionError(f"Installed filter expected 1 visible card, got {installed_count}.")
        page.evaluate("window.NexusCivitai.close()")
        page.wait_for_timeout(150)
        page.evaluate("window.NexusCivitai.open()")
        page.wait_for_timeout(300)
        if page.locator("#civitaiInstalledToggle").is_checked():
            raise AssertionError("Installed filter stayed checked after modal reopen.")
        if not page.locator("#civitaiSearchGrid article").count():
            raise AssertionError("Civitai modal reopened into an empty cached filter state.")
        page.uncheck("#civitaiInstalledToggle")
        page.wait_for_timeout(250)

        page.fill("#civitaiUrlInput", "#style")
        page.press("#civitaiUrlInput", "Enter")
        page.wait_for_function(
            "() => window.__civitaiSmokeRequests ? window.__civitaiSmokeRequests >= 0 : true",
            timeout=1000,
        )
        page.wait_for_timeout(250)
        style_payload = captured[-1]
        if style_payload.get("query") or style_payload.get("tag") != "style":
            raise AssertionError(f"Hash tag payload is wrong: {style_payload}")

        page.evaluate("window.NexusCivitai.setType('LORA')")
        page.wait_for_timeout(250)
        lora_payload = captured[-1]
        if lora_payload.get("types") != "LORA":
            raise AssertionError(f"Type payload is wrong: {lora_payload}")
        if not page.locator("#civitaiTagChips button", has_text="#character").count():
            raise AssertionError("LoRA tag chips did not render real Civitai tags.")

        page.evaluate("window.NexusCivitai.setTag('character')")
        page.wait_for_timeout(250)
        tag_payload = captured[-1]
        if tag_payload.get("tag") != "character":
            raise AssertionError(f"Chip tag payload is wrong: {tag_payload}")

        page.fill("#civitaiUrlInput", "splash art")
        page.press("#civitaiUrlInput", "Enter")
        page.wait_for_timeout(300)
        text_payload = captured[-1]
        if text_payload.get("query") != "splash art" or text_payload.get("tag"):
            raise AssertionError(f"Text search should send query and clear tag, got {text_payload}")
        if page.locator("#civitaiSearchGrid article").count() != 1:
            raise AssertionError("Text search did not locally filter results inside the selected type/base filters.")
        if not page.locator("#civitaiSearchGrid article", has_text="Splash Art Contract").count():
            raise AssertionError("Text search did not show the matching Splash Art result.")
        page.evaluate(
            """async () => {
                document.getElementById('civitaiUrlInput').value = '';
                document.getElementById('civitaiTypeFilter').value = '';
                document.getElementById('civitaiTagFilter').value = '';
                document.getElementById('civitaiBaseModelFilter').value = '';
                await window.NexusCivitai.search();
            }"""
        )

        before = page.locator("#civitaiSearchGrid article").count()
        page.evaluate("window.NexusCivitai.loadMore()")
        after = page.locator("#civitaiSearchGrid article").count()
        if after <= before:
            raise AssertionError(f"Lazy load did not append cards: before={before} after={after}")
        if not captured[-1].get("cursor"):
            raise AssertionError(f"Lazy load did not send cursor: {captured[-1]}")

        browser.close()

    print("civitai modal contract smoke ok")


if __name__ == "__main__":
    main()
