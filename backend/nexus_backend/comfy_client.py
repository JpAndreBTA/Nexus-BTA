from __future__ import annotations

import asyncio
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .config import NexusSettings, runtime_python


class ComfyExecutionError(RuntimeError):
    pass


class ComfyClient:
    def __init__(self, settings: NexusSettings):
        self.settings = settings
        self.process: subprocess.Popen[str] | None = None

    @property
    def base_url(self) -> str:
        runtime = self.settings.runtime
        return f"http://{runtime.host}:{runtime.comfy_port}"

    async def is_running(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=1.5) as client:
                response = await client.get(f"{self.base_url}/system_stats")
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    async def ensure_running(self) -> None:
        if await self.is_running():
            return
        self.start()
        deadline = time.time() + 120
        while time.time() < deadline:
            if await self.is_running():
                return
            await asyncio.sleep(1)
        raise RuntimeError("ComfyUI embedded runtime did not become ready in time.")

    async def restart(self) -> None:
        self.stop()
        await asyncio.sleep(1)
        await self.ensure_running()

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            self.process = None
            return
        self.process.terminate()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        finally:
            self.process = None

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        if not self.settings.comfy_root.exists():
            raise FileNotFoundError(
                f"Embedded ComfyUI runtime not found at {self.settings.comfy_root}. "
                "Run scripts/bootstrap_nexus_runtime.ps1 first."
            )

        python_exe = runtime_python(self.settings)
        main_py = self.settings.comfy_root / "main.py"
        if not main_py.exists():
            raise FileNotFoundError(f"ComfyUI main.py not found: {main_py}")

        database_path = self.settings.user_dir / "comfyui.db"
        database_path.parent.mkdir(parents=True, exist_ok=True)
        extra_model_paths = self.settings.project_root / "config" / "nexus_extra_model_paths.yaml"

        args = [
            str(python_exe),
            str(main_py),
            "--listen",
            self.settings.runtime.host,
            "--port",
            str(self.settings.runtime.comfy_port),
            "--base-directory",
            str(self.settings.project_root),
            "--output-directory",
            str(self.settings.output_dir),
            "--input-directory",
            str(self.settings.input_dir),
            "--temp-directory",
            str(self.settings.temp_dir),
            "--user-directory",
            str(self.settings.user_dir),
            "--database-url",
            f"sqlite:///{database_path.as_posix()}",
            "--disable-auto-launch",
            "--log-stdout",
            "--enable-cors-header",
            "*",
        ]
        if extra_model_paths.exists():
            args.extend(["--extra-model-paths-config", str(extra_model_paths)])
        args.extend(self._runtime_flags())

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"
        env["NEXUS_BTA"] = "1"

        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        self.process = subprocess.Popen(
            args,
            cwd=str(self.settings.comfy_root),
            env=env,
            text=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            startupinfo=startupinfo,
        )

    def _runtime_flags(self) -> list[str]:
        runtime = self.settings.runtime
        flags: list[str] = []
        vram = runtime.vram_policy.lower()
        if vram in {"low", "lowvram"}:
            flags.append("--lowvram")
        elif vram in {"med", "normal", "balanced"}:
            flags.append("--normalvram")
        elif vram in {"high", "highvram"}:
            flags.append("--highvram")
        elif vram == "cpu":
            flags.append("--cpu")

        precision = runtime.precision.lower()
        if precision in {"fp16", "fp32"}:
            flags.append(f"--force-{precision}")

        if runtime.disable_xformers:
            flags.append("--disable-xformers")
        attention = runtime.attention_backend.lower()
        if attention == "sage" or (attention == "auto" and runtime.enable_sage_attention):
            flags.append("--use-sage-attention")
        elif attention == "flash" or runtime.enable_flash_attention:
            flags.append("--use-flash-attention")
        elif attention == "pytorch":
            flags.append("--use-pytorch-cross-attention")
        return flags

    async def object_info(self) -> dict[str, Any]:
        await self.ensure_running()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/object_info")
            response.raise_for_status()
            return response.json()

    async def queue_prompt(self, workflow: dict[str, Any], client_id: str | None = None) -> str:
        await self.ensure_running()
        payload = {"prompt": workflow, "client_id": client_id or str(uuid.uuid4())}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(f"{self.base_url}/prompt", json=payload)
            data = _response_json(response)
            if response.status_code >= 400:
                raise RuntimeError(_format_comfy_error(data, response.text))
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise RuntimeError(f"ComfyUI did not return prompt_id: {data}")
        return str(prompt_id)

    async def run_workflow(
        self,
        workflow: dict[str, Any],
        progress_callback: Any | None = None,
        timeout_seconds: int = 3600,
    ) -> tuple[str, list[dict[str, Any]]]:
        client_id = str(uuid.uuid4())
        prompt_id = await self.queue_prompt(workflow, client_id=client_id)
        if progress_callback:
            progress_callback({"status": "queued", "progress": 4, "message": f"Queued {prompt_id}", "prompt_id": prompt_id})
        try:
            await self.watch_prompt_progress(prompt_id, client_id, progress_callback, timeout_seconds=timeout_seconds)
        except ComfyExecutionError:
            raise
        except Exception as exc:
            if progress_callback:
                progress_callback({"status": "polling", "progress": 12, "message": f"Progress stream unavailable, polling history: {exc}"})
        outputs = await self.wait_for_outputs(prompt_id, timeout_seconds=timeout_seconds)
        if progress_callback:
            progress_callback({"status": "completed", "progress": 100, "message": "Generation completed.", "prompt_id": prompt_id})
        return prompt_id, outputs

    async def watch_prompt_progress(
        self,
        prompt_id: str,
        client_id: str,
        progress_callback: Any | None = None,
        timeout_seconds: int = 3600,
    ) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("websockets package is not installed") from exc

        parsed = urlparse(self.base_url)
        scheme = "wss" if parsed.scheme == "https" else "ws"
        ws_url = f"{scheme}://{parsed.netloc}/ws?clientId={quote(client_id)}"
        deadline = time.time() + timeout_seconds
        async with websockets.connect(ws_url, max_size=None) as websocket:
            while time.time() < deadline:
                message = await asyncio.wait_for(websocket.recv(), timeout=min(30, max(1, deadline - time.time())))
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                event_type = event.get("type")
                data = event.get("data") or {}
                event_prompt = data.get("prompt_id")
                if event_prompt and str(event_prompt) != str(prompt_id):
                    continue

                if event_type == "progress":
                    value = float(data.get("value") or 0)
                    maximum = max(1.0, float(data.get("max") or 1))
                    if progress_callback:
                        progress_callback(
                            {
                                "status": "running",
                                "progress": min(99, max(5, round((value / maximum) * 100))),
                                "current_step": int(value),
                                "total_steps": int(maximum),
                                "node": str(data.get("node") or ""),
                                "message": f"Step {int(value)}/{int(maximum)}",
                                "prompt_id": prompt_id,
                            }
                        )
                elif event_type == "executing":
                    node = data.get("node")
                    if node is None and str(data.get("prompt_id") or prompt_id) == str(prompt_id):
                        return
                    if progress_callback:
                        progress_callback(
                            {
                                "status": "running",
                                "progress": 8,
                                "node": str(node or ""),
                                "message": f"Executing node {node}",
                                "prompt_id": prompt_id,
                            }
                        )
                elif event_type in {"execution_error", "execution_interrupted"}:
                    message_text = str(data.get("exception_message") or data.get("exception_type") or event_type)
                    if progress_callback:
                        progress_callback({"status": "failed", "progress": 100, "message": message_text, "prompt_id": prompt_id})
                    raise ComfyExecutionError(message_text)

    async def history(self, prompt_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()

    async def wait_for_outputs(self, prompt_id: str, timeout_seconds: int = 3600) -> list[dict[str, Any]]:
        deadline = time.time() + timeout_seconds
        while time.time() < deadline:
            history = await self.history(prompt_id)
            item = history.get(prompt_id)
            if item:
                outputs = extract_outputs(item)
                if outputs:
                    return outputs
                status = item.get("status", {})
                if status.get("status_str") == "error":
                    raise ComfyExecutionError(_history_error_message(status))
                if item.get("status", {}).get("completed"):
                    return []
            await asyncio.sleep(1)
        raise TimeoutError(f"ComfyUI job timed out: {prompt_id}")


def extract_outputs(history_item: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for node_output in history_item.get("outputs", {}).values():
        for key in ["images", "videos", "gifs"]:
            for item in node_output.get(key, []) or []:
                filename = item.get("filename")
                if not filename:
                    continue
                subfolder = item.get("subfolder", "")
                output_type = item.get("type", "output")
                if output_type != "output":
                    continue
                relative_path = f"{subfolder.strip('/')}/{filename}" if subfolder else filename
                safe_relative_path = relative_path.replace("\\", "/")
                url = f"/outputs/{quote(safe_relative_path, safe='/')}"
                suffix = Path(filename).suffix.lower()
                media_kind = "video" if suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"} else key[:-1]
                outputs.append(
                    {
                        "kind": media_kind,
                        "filename": filename,
                        "subfolder": subfolder,
                        "type": output_type,
                        "path": relative_path,
                        "url": url,
                    }
                )
    return outputs


def _response_json(response: httpx.Response) -> dict[str, Any]:
    try:
        data = response.json()
        return data if isinstance(data, dict) else {"response": data}
    except ValueError:
        return {}


def _format_comfy_error(data: dict[str, Any], fallback: str) -> str:
    if not data:
        return fallback or "ComfyUI returned an empty error response."

    parts: list[str] = []
    error = data.get("error")
    if isinstance(error, dict):
        message = error.get("message") or error.get("details") or error.get("type")
        if message:
            parts.append(str(message))
    elif error:
        parts.append(str(error))

    node_errors = data.get("node_errors")
    if isinstance(node_errors, dict) and node_errors:
        details: list[str] = []
        for node_id, node_error in list(node_errors.items())[:6]:
            if isinstance(node_error, dict):
                class_type = node_error.get("class_type") or node_error.get("type") or "node"
                errors = node_error.get("errors") or []
                detail = node_error.get("message") or "validation failed"
                if isinstance(errors, list) and errors:
                    first = errors[0]
                    if isinstance(first, dict):
                        detail = first.get("message") or first.get("details") or first.get("type") or detail
                    else:
                        detail = str(first)
                details.append(f"{node_id} {class_type}: {detail}")
            else:
                details.append(f"{node_id}: {node_error}")
        parts.append("Node errors: " + " | ".join(details))

    return " ".join(parts) or str(data)


def _history_error_message(status: dict[str, Any]) -> str:
    for event_type, data in reversed(status.get("messages") or []):
        if event_type == "execution_error" and isinstance(data, dict):
            return str(data.get("exception_message") or data.get("exception_type") or "ComfyUI execution failed.")
    return "ComfyUI execution failed."
