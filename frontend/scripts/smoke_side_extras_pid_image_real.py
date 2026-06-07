import json
import shutil
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "test-results"
RESULTS.mkdir(exist_ok=True)
BASE = "http://127.0.0.1:7861/ui"
SOURCE_IMAGE = ROOT / "input" / "Smoke_splashART.jpeg"
PROBE_IMAGE = ROOT / "output" / "image" / "side_extras_pid_probe.jpeg"


def ensure_probe_image() -> None:
    if not SOURCE_IMAGE.exists():
        raise AssertionError(f"Missing Smoke_splashART.jpeg: {SOURCE_IMAGE}")
    PROBE_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(SOURCE_IMAGE, PROBE_IMAGE)


def analyze_image(path: Path, label: str) -> dict[str, object]:
    if not path.exists():
        raise AssertionError(f"{label}: missing image {path}")
    image = Image.open(path).convert("RGB")
    arr = np.asarray(image.resize((256, 128)), dtype=np.float32)
    local_diff = float((np.abs(np.diff(arr, axis=0)).mean() + np.abs(np.diff(arr, axis=1)).mean()) / 2)
    metrics = {
        "path": str(path),
        "width": image.width,
        "height": image.height,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "local_diff": local_diff,
    }
    if metrics["std"] < 2.0:
        raise AssertionError(f"{label}: image appears blank/flat: {metrics!r}")
    if metrics["local_diff"] > 60:
        raise AssertionError(f"{label}: image looks noise-heavy: {metrics!r}")
    return metrics


def output_path_from_url(url: str) -> Path:
    if "/outputs/" not in url:
        raise AssertionError(f"Unexpected output URL: {url}")
    rel = url.split("/outputs/", 1)[1].split("?", 1)[0]
    return ROOT / "output" / Path(rel.replace("/", "\\"))


def make_comparison(source: Path, output: Path) -> Path:
    sheet = RESULTS / "front_side_extras_pid_image_compare.jpg"
    src = Image.open(source).convert("RGB")
    out = Image.open(output).convert("RGB")
    src.thumbnail((480, 260))
    out.thumbnail((480, 260))
    canvas = Image.new("RGB", (max(src.width, out.width), src.height + out.height), (8, 8, 10))
    canvas.paste(src, (0, 0))
    canvas.paste(out, (0, src.height))
    canvas.save(sheet, quality=92)
    return sheet


def metadata_for(output_path: Path) -> dict[str, object]:
    meta_path = output_path.with_name(output_path.name + ".nexus.json")
    if not meta_path.exists():
        raise AssertionError(f"Missing Extras metadata sidecar: {meta_path}")
    return json.loads(meta_path.read_text(encoding="utf-8"))


def main() -> None:
    ensure_probe_image()
    source_metrics = analyze_image(PROBE_IMAGE, "pid_source")
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1440, "height": 920})
        page.set_default_timeout(1800000)
        page.on("dialog", lambda dialog: dialog.accept())
        page.goto(f"{BASE}?side_pid_image_smoke={int(time.time())}", wait_until="networkidle", timeout=60000)
        page.wait_for_function("() => document.querySelector('#appBootOverlay')?.classList.contains('hidden')", timeout=120000)
        page.wait_for_function("() => backendOnline === true && typeof maybeRunPostGenerationExtras === 'function'", timeout=120000)
        page.evaluate(
            """
            async () => {
              document.querySelector('#extrasPostEnableToggle').checked = true;
              document.querySelector('#extrasPostUpscaleEngine').value = 'nvidia_pid';
              document.querySelector('#extrasPostScale').value = '2x';
              document.querySelector('#extrasPostRestoreAlignToggle').checked = true;
              document.querySelector('#sideDenoiseToggle').checked = true;
              document.querySelector('#sideDenoiseStrengthSlider').value = '0.18';
              document.querySelector('#sideDetailToggle').checked = true;
              document.querySelector('#sideDetailStrengthSlider').value = '0.24';
              document.querySelector('#sideFaceRestoreToggle').checked = false;
              const profile = document.querySelector('#extrasPidProfileSelect');
              if (profile) {
                profile.value = 'lowvram_zimage_2k';
                syncExtrasPidProfileFields();
              }
              syncSideExtrasPostUi({ prompt: false });
              syncSideRefineUi({ prompt: false });
              const ready = await ensureSideExtrasPostReady();
              if (!ready) throw new Error('Side PiD prerequisites were rejected.');
            }
            """
        )
        result = page.evaluate(
            """
            async () => {
              return await maybeRunPostGenerationExtras({
                outputs: [{
                  kind: 'image',
                  media_type: 'image',
                  filename: 'side_extras_pid_probe.jpeg',
                  url: '/outputs/image/side_extras_pid_probe.jpeg'
                }]
              }, {
                prompt: 'Smoke splash art, clean details, low noise, high quality restored image'
              });
            }
            """
        )
        page.screenshot(path=str(RESULTS / "front_side_extras_pid_image_real.png"), full_page=True)
        browser.close()

    outputs = result.get("outputs") or []
    if not outputs:
        raise AssertionError(f"Side PiD image did not return outputs: {result!r}")
    output_path = output_path_from_url(outputs[0]["url"])
    output_metrics = analyze_image(output_path, "pid_output")
    metadata = metadata_for(output_path)
    upscale = metadata.get("extras", {}).get("upscale", {})
    if upscale.get("runtime_engine") != "nvidia_pid":
        raise AssertionError(f"Expected real NVIDIA PiD runtime, got {upscale!r}")
    if "PiD" not in str(upscale.get("workflow_reference") or "") or upscale.get("latent_decode") is not True:
        raise AssertionError(f"Expected staged PiD latent output, got {upscale!r}")
    sheet = make_comparison(PROBE_IMAGE, output_path)
    report = {
        "source": str(PROBE_IMAGE),
        "output": str(output_path),
        "comparison_sheet": str(sheet),
        "source_metrics": source_metrics,
        "output_metrics": output_metrics,
        "metadata": {
            "upscale": upscale,
            "denoise": metadata.get("extras", {}).get("denoise", {}),
            "detail_refine": metadata.get("extras", {}).get("detail_refine", {}),
            "face_restore": metadata.get("extras", {}).get("face_restore", {}),
        },
        "front_response": result,
    }
    (RESULTS / "front_side_extras_pid_image_real.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"ok side PiD image post-process battery: {output_path}")


if __name__ == "__main__":
    main()
