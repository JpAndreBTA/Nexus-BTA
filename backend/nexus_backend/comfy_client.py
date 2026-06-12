from __future__ import annotations

import asyncio
import importlib
import importlib.util
import json
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse

import httpx

from .config import NexusSettings, runtime_python


def _xformers_cuda_probe_error() -> str:
    try:
        import torch  # type: ignore
        import xformers.ops as xops  # type: ignore

        if importlib.util.find_spec("xformers._C") is None:
            return "xformers._C extension is not available"
        if not torch.cuda.is_available():
            return "CUDA is not available for xFormers"

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        q = torch.randn((1, 32, 30, 128), device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        xops.memory_efficient_attention(q, k, v)
        torch.cuda.synchronize()
        return ""
    except Exception as exc:
        return f"{type(exc).__name__}: {str(exc)[:220]}"


def _module_available(module_name: str) -> bool:
    try:
        if module_name == "xformers":
            return _xformers_cuda_probe_error() == ""
        if importlib.util.find_spec(module_name) is None:
            return False
        if module_name == "sageattention" and importlib.util.find_spec("triton") is None:
            return False
        if module_name in {"sageattention", "flash_attn"}:
            importlib.import_module(module_name)
            if module_name == "sageattention":
                importlib.import_module("triton")
        return True
    except Exception:
        return False


class ComfyExecutionError(RuntimeError):
    pass


def _workflow_required_classes(workflow: dict[str, Any]) -> list[tuple[str, str]]:
    required: list[tuple[str, str]] = []
    for node in (workflow or {}).values():
        if not isinstance(node, dict):
            continue
        class_type = str(node.get("class_type") or "").strip()
        if not class_type:
            continue
        title = ""
        meta = node.get("_meta")
        if isinstance(meta, dict):
            title = str(meta.get("title") or "").strip()
        required.append((class_type, title))
    return required


def _format_missing_workflow_nodes(
    workflow: dict[str, Any],
    object_info: dict[str, Any],
    *,
    custom_nodes_dir: Path,
    comfy_root: Path,
) -> str:
    available = set(object_info or {})
    missing: dict[str, set[str]] = {}
    for class_type, title in _workflow_required_classes(workflow):
        if class_type not in available:
            missing.setdefault(class_type, set())
            if title:
                missing[class_type].add(title)
    if not missing:
        return ""

    parts: list[str] = []
    for class_type, titles in sorted(missing.items()):
        if titles:
            parts.append(f"{class_type} ({', '.join(sorted(titles)[:3])})")
        else:
            parts.append(class_type)
    shown = ", ".join(parts[:16])
    if len(parts) > 16:
        shown += f", +{len(parts) - 16} more"

    hints: list[str] = []
    if "LTXVTiledVAEDecode" in missing or "LTXVAudioVAEDecode" in missing:
        hints.append(
            "LTX Decode Frames/Audio Decode comes from ComfyUI-LTXVideo. "
            "If the folder is already installed, ComfyUI is still failing to import/register it in the configured runtime; "
            "run update.bat or restart with run.bat to force dependency/import repair."
        )
    if {"LoadVideo", "GetVideoComponents", "CreateVideo", "SaveVideo", "VHS_LoadVideo", "VHS_VideoCombine"} & set(missing):
        hints.append("Video load/save/combine nodes require the video helper custom nodes installed in the active custom_nodes folder.")
    if {"VAEDecode", "VAEEncode", "VAELoader"} & set(missing):
        hints.append("Core VAE nodes are missing from ComfyUI's registry; check ComfyUI startup/import errors.")

    message = (
        "ComfyUI did not load required workflow nodes: "
        f"{shown}. Active custom_nodes folder: {custom_nodes_dir}. Comfy root: {comfy_root}."
    )
    if hints:
        message += " " + " ".join(hints)
    return message


class ComfyClient:
    def __init__(self, settings: NexusSettings):
        self.settings = settings
        self.process: subprocess.Popen[str] | None = None
        self._log_handle: Any | None = None
        self._started_runtime_signature: str = ""
        self._start_lock = asyncio.Lock()
        self._owned_external_pid: int | None = None
        self._attention_backend_override: str | None = None

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
            if not self._adopt_running_runtime():
                owner = self.runtime_owner_description()
                raise RuntimeError(f"Port {self.settings.runtime.comfy_port} is already used by another ComfyUI/runtime process: {owner}")
            return
        async with self._start_lock:
            if await self.is_running():
                if not self._adopt_running_runtime():
                    owner = self.runtime_owner_description()
                    raise RuntimeError(f"Port {self.settings.runtime.comfy_port} is already used by another ComfyUI/runtime process: {owner}")
                return
            self.start()
            start_timeout = max(60.0, float(os.environ.get("NEXUS_COMFY_START_TIMEOUT", "300")))
            deadline = time.time() + start_timeout
            while time.time() < deadline:
                if await self.is_running():
                    return
                await asyncio.sleep(1)
            raise RuntimeError(f"ComfyUI embedded runtime did not become ready within {int(start_timeout)} seconds.")

    async def start_nowait(self) -> dict[str, Any]:
        if await self.is_running():
            if not self._adopt_running_runtime():
                owner = self.runtime_owner_description()
                raise RuntimeError(f"Port {self.settings.runtime.comfy_port} is already used by another ComfyUI/runtime process: {owner}")
            return {"status": "running", "url": self.base_url}
        async with self._start_lock:
            if await self.is_running():
                if not self._adopt_running_runtime():
                    owner = self.runtime_owner_description()
                    raise RuntimeError(f"Port {self.settings.runtime.comfy_port} is already used by another ComfyUI/runtime process: {owner}")
                return {"status": "running", "url": self.base_url}
            self.start()
            return {"status": "starting", "url": self.base_url, "log": str(self.settings.project_root / "logs" / "comfyui.log")}

    async def restart(self) -> None:
        self.stop()
        await asyncio.sleep(1)
        await self.ensure_running()

    def use_preset_attention_backend(self, preset: str) -> bool:
        requested = str(preset or "").strip().lower()
        desired: str | None = None
        if requested == "qwen" and self._default_uses_sage_attention():
            desired = "xformers" if self._xformers_allowed() else "pytorch"
        if desired == self._attention_backend_override:
            return False
        self._attention_backend_override = desired
        return True

    def stop(self) -> None:
        if not self.process or self.process.poll() is not None:
            self.process = None
            self._stop_owned_external_runtime()
            self._close_log_handle()
            return
        pid = self.process.pid
        self._stop_process_tree(pid)
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=10)
        finally:
            self.process = None
            self._owned_external_pid = None
            self._close_log_handle()

    def start(self) -> None:
        if self.process and self.process.poll() is None:
            return
        self._close_log_handle()
        owner_pid = self._runtime_owner_pid()
        if owner_pid:
            if self._adopt_running_runtime():
                return
            raise RuntimeError(f"Port {self.settings.runtime.comfy_port} is already used by another runtime: {self.runtime_owner_description(owner_pid)}")
        if self._owned_external_pid:
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
        comfy_base_dir = self._comfy_base_directory()

        args = [
            str(python_exe),
            str(main_py),
            "--listen",
            self.settings.runtime.host,
            "--port",
            str(self.settings.runtime.comfy_port),
            "--base-directory",
            str(comfy_base_dir),
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
        env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True,garbage_collection_threshold:0.65")
        env.setdefault("SPARSE_CONV_BACKEND", "flex_gemm")
        flex_chunk_mb, flex_offload_mb = self._trellis_flex_gemm_profile()
        env.setdefault("NEXUS_TRELLIS_FLEX_GEMM_CHUNK_MB", str(flex_chunk_mb))
        env.setdefault("NEXUS_TRELLIS_FLEX_GEMM_OFFLOAD_OUTPUT_MB", str(flex_offload_mb))

        startupinfo = None
        if os.name == "nt":
            startupinfo = subprocess.STARTUPINFO()
            startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = 0

        log_path = self.settings.project_root / "logs" / "comfyui.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_handle = log_path.open("a", encoding="utf-8", errors="replace")
        self._log_handle.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] Starting ComfyUI: {' '.join(args)}\n")
        self._log_handle.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Nexus custom nodes: {self.settings.custom_nodes_dir}\n")
        self._log_handle.flush()
        try:
            self.process = subprocess.Popen(
                args,
                cwd=str(self.settings.comfy_root),
                env=env,
                text=True,
                stdout=self._log_handle,
                stderr=subprocess.STDOUT,
                startupinfo=startupinfo,
            )
        except Exception:
            self._close_log_handle()
            raise
        self._started_runtime_signature = self.runtime_signature()

    def _comfy_base_directory(self) -> Path:
        custom_nodes_dir = self.settings.custom_nodes_dir
        try:
            if custom_nodes_dir.name.lower() == "custom_nodes":
                custom_nodes_dir.mkdir(parents=True, exist_ok=True)
                return custom_nodes_dir.parent
        except Exception:
            pass

        embedded_comfy_root = self.settings.project_root / "runtime" / "ComfyUI"
        try:
            use_project_base = self.settings.comfy_root.resolve() == embedded_comfy_root.resolve()
        except OSError:
            use_project_base = str(self.settings.comfy_root).strip().rstrip("\\/").lower() == str(embedded_comfy_root).strip().rstrip("\\/").lower()
        return self.settings.project_root if use_project_base else self.settings.comfy_root

    def _close_log_handle(self) -> None:
        handle = self._log_handle
        self._log_handle = None
        if handle:
            try:
                handle.close()
            except Exception:
                pass

    def _runtime_owner_pid(self) -> int | None:
        try:
            import psutil

            for conn in psutil.net_connections(kind="inet"):
                if conn.status == "LISTEN" and conn.laddr and getattr(conn.laddr, "port", None) == self.settings.runtime.comfy_port:
                    return conn.pid
        except Exception:
            return None
        return None

    def _pid_belongs_to_runtime(self, pid: int | None) -> bool:
        if not pid:
            return False
        if self.process and self.process.pid == pid:
            return True
        try:
            import psutil

            process = psutil.Process(pid)
            command = " ".join(process.cmdline())
            project_root = str(self.settings.project_root).replace("/", "\\").lower()
            comfy_root = str(self.settings.comfy_root).replace("/", "\\").lower()
            normalized = command.replace("/", "\\").lower()
            main_py = str(self.settings.comfy_root / "main.py").replace("/", "\\").lower()
            return (project_root in normalized or comfy_root in normalized or main_py in normalized) and "main.py" in normalized
        except Exception:
            return False

    def _adopt_running_runtime(self) -> bool:
        pid = self._runtime_owner_pid()
        if self._pid_belongs_to_runtime(pid):
            self._owned_external_pid = pid
            self._started_runtime_signature = self.runtime_signature()
            return True
        return False

    def runtime_owner_description(self, pid: int | None = None) -> str:
        pid = pid or self._runtime_owner_pid()
        if not pid:
            return "unknown process"
        try:
            import psutil

            process = psutil.Process(pid)
            command = " ".join(process.cmdline())
            return f"pid={pid} command={command}"
        except Exception:
            return f"pid={pid}"

    def _stop_process_tree(self, pid: int | None) -> None:
        if not pid:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception:
            pass

    def _stop_owned_external_runtime(self) -> None:
        pid = self._owned_external_pid or self._runtime_owner_pid()
        if not self._pid_belongs_to_runtime(pid):
            self._owned_external_pid = None
            return
        self._stop_process_tree(pid)
        self._owned_external_pid = None

    def runtime_signature(self) -> str:
        runtime = self.settings.runtime
        payload = {
            "comfy_python": str(self.settings.comfy_python.resolve() if self.settings.comfy_python.exists() else self.settings.comfy_python),
            "comfy_root": str(self.settings.comfy_root.resolve() if self.settings.comfy_root.exists() else self.settings.comfy_root),
            "custom_nodes_dir": str(self.settings.custom_nodes_dir.resolve() if self.settings.custom_nodes_dir.exists() else self.settings.custom_nodes_dir),
            "vram_policy": runtime.vram_policy.lower(),
            "gpu_memory_gb": runtime.gpu_memory_gb,
            "precision": runtime.precision.lower(),
            "disable_xformers": bool(runtime.disable_xformers),
            "attention_backend": runtime.attention_backend.lower(),
            "attention_backend_override": self._attention_backend_override or "",
            "enable_sage_attention": bool(runtime.enable_sage_attention),
            "enable_flash_attention": bool(runtime.enable_flash_attention),
            "enable_pytorch_attention": bool(getattr(runtime, "enable_pytorch_attention", True)),
        }
        return json.dumps(payload, sort_keys=True)

    def runtime_changed_since_start(self) -> bool:
        owned_running = bool(self.process and self.process.poll() is None) or bool(self._owned_external_pid and self._pid_belongs_to_runtime(self._owned_external_pid))
        return bool(owned_running and self._started_runtime_signature and self._started_runtime_signature != self.runtime_signature())

    def _runtime_flags(self) -> list[str]:
        runtime = self.settings.runtime
        flags: list[str] = []
        vram = runtime.vram_policy.lower().replace(" ", "").replace("_", "").replace("-", "")
        if vram in {"gpu", "gpuonly", "onlygpu", "cudaonly"}:
            flags.append("--gpu-only")
        elif vram in {"shared", "vramshared", "sharedvram", "dynamic", "default", "auto", "low", "lowvram", "med", "medium", "medvram", "normal", "balanced", "balance", "high", "highvram"}:
            flags.append("--enable-dynamic-vram")
        elif vram in {"low", "lowvram"}:
            flags.append("--lowvram")
        elif vram in {"med", "medium", "medvram", "normal", "balanced"}:
            flags.append("--normalvram")
        elif vram in {"high", "highvram"}:
            flags.append("--highvram")
        elif vram.startswith("cpu"):
            flags.append("--cpu")

        reserve_vram = self._reserve_vram_gb()
        if reserve_vram is not None:
            flags.extend(["--reserve-vram", f"{reserve_vram:.2f}"])

        precision = runtime.precision.lower().replace("_", "-")
        if precision in {"fp16", "fp32"}:
            flags.append(f"--force-{precision}")
        elif precision == "bf16":
            flags.extend(["--force-fp16", "--bf16-unet"])
        elif precision == "fp8":
            flags.append("--fp8_e4m3fn-unet")

        attention = (self._attention_backend_override or runtime.attention_backend).lower().replace(" ", "").replace("_", "").replace("-", "")
        if attention in {"xformers", "xformer"}:
            if not _module_available("xformers"):
                raise RuntimeError(
                    "xFormers is selected for the Nexus runtime, but the configured Comfy Python cannot import xformers. "
                    "Run update.bat to install the current runtime requirements, then restart Nexus."
                )
        elif runtime.disable_xformers or not _module_available("xformers"):
            flags.append("--disable-xformers")

        if (
            (attention in {"sage", "sageattention"} or (attention == "auto" and runtime.enable_sage_attention))
        ):
            if not _module_available("sageattention"):
                raise RuntimeError(
                    "SageAttention is enabled for the Nexus runtime, but the configured Comfy Python cannot import "
                    "sageattention with its Triton runtime. Run update.bat to install the current runtime requirements, "
                    "then restart Nexus."
                )
            flags.append("--use-sage-attention")
        elif (attention in {"flash", "flashattention"} or runtime.enable_flash_attention) and _module_available("flash_attn"):
            flags.append("--use-flash-attention")
        elif attention in {"pytorch", "pytorchsdpa", "sdpa"}:
            flags.append("--use-pytorch-cross-attention")
        elif not bool(getattr(runtime, "enable_pytorch_attention", True)) and (runtime.disable_xformers or not _module_available("xformers")):
            flags.append("--use-split-cross-attention")
        return flags

    def _xformers_allowed(self) -> bool:
        return not bool(self.settings.runtime.disable_xformers) and _module_available("xformers")

    def _default_uses_sage_attention(self) -> bool:
        runtime = self.settings.runtime
        attention = runtime.attention_backend.lower().replace(" ", "").replace("_", "").replace("-", "")
        return attention in {"sage", "sageattention"} or (attention == "auto" and bool(runtime.enable_sage_attention))

    def _reserve_vram_gb(self) -> float | None:
        try:
            requested = float(self.settings.runtime.gpu_memory_gb or 0)
        except (TypeError, ValueError):
            return None
        if requested <= 0:
            return None
        total = self._cuda_total_vram_gb()
        if total is None or total <= 0:
            return None
        reserve = max(0.25, total - requested)
        if reserve >= total:
            reserve = max(0.25, total - 1.0)
        return round(reserve, 2)

    def _trellis_flex_gemm_profile(self) -> tuple[int, int]:
        try:
            requested = float(self.settings.runtime.gpu_memory_gb or 0)
        except (TypeError, ValueError):
            requested = 0
        if requested and requested <= 8:
            return (128, 768)
        if requested and requested <= 12:
            return (256, 1536)
        return (384, 2048)

    @staticmethod
    def _cuda_total_vram_gb() -> float | None:
        try:
            import torch  # type: ignore

            if not torch.cuda.is_available():
                return None
            props = torch.cuda.get_device_properties(0)
            return float(props.total_memory) / (1024 ** 3)
        except Exception:
            return None

    async def object_info(self) -> dict[str, Any]:
        await self.ensure_running()
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/object_info")
            response.raise_for_status()
            data = response.json()
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    data = {}
            return data if isinstance(data, dict) else {}

    async def system_stats(self) -> dict[str, Any]:
        if not await self.is_running():
            return {}
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{self.base_url}/system_stats")
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}

    async def free_memory(self, unload_models: bool = True, free_memory: bool = True) -> dict[str, Any]:
        if not await self.is_running():
            return {"status": "stopped", "url": self.base_url}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(
                f"{self.base_url}/free",
                json={"unload_models": unload_models, "free_memory": free_memory},
            )
            response.raise_for_status()
        return {"status": "free_requested", "url": self.base_url, "unload_models": unload_models, "free_memory": free_memory}

    async def interrupt(self, prompt_id: str | None = None) -> dict[str, Any]:
        if not await self.is_running():
            return {"status": "stopped", "url": self.base_url}
        payload = {"prompt_id": prompt_id} if prompt_id else {}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{self.base_url}/interrupt", json=payload)
            response.raise_for_status()
        return {"status": "interrupt_requested", "url": self.base_url, "prompt_id": prompt_id}

    async def clear_queue(self) -> dict[str, Any]:
        if not await self.is_running():
            return {"status": "stopped", "url": self.base_url}
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.post(f"{self.base_url}/queue", json={"clear": True})
            response.raise_for_status()
        return {"status": "queue_cleared", "url": self.base_url}

    async def queue_prompt(self, workflow: dict[str, Any], client_id: str | None = None) -> str:
        await self.ensure_running()
        object_info = await self.object_info()
        missing_message = _format_missing_workflow_nodes(
            workflow,
            object_info,
            custom_nodes_dir=self.settings.custom_nodes_dir,
            comfy_root=self.settings.comfy_root,
        )
        if missing_message:
            raise RuntimeError(missing_message)
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
        started_at = time.time()
        prompt_id = await self.queue_prompt(workflow, client_id=client_id)
        if progress_callback:
            progress_callback({"status": "queued", "progress": 10, "message": "Queued in ComfyUI", "prompt_id": prompt_id})
        try:
            await self.watch_prompt_progress(prompt_id, client_id, progress_callback, timeout_seconds=timeout_seconds)
        except (TimeoutError, asyncio.TimeoutError):
            await self.interrupt(prompt_id)
            raise TimeoutError(f"ComfyUI job timed out after {timeout_seconds}s: {prompt_id}")
        except ComfyExecutionError:
            raise
        except Exception as exc:
            if progress_callback:
                progress_callback(
                    {
                        "status": "polling",
                        "progress": 12,
                        "message": "Syncing ComfyUI history",
                        "detail": str(exc),
                        "elapsed_seconds": round(time.time() - started_at, 1),
                        "prompt_id": prompt_id,
                    }
                )
        try:
            outputs = await self.wait_for_outputs(
                prompt_id,
                timeout_seconds=timeout_seconds,
                progress_callback=progress_callback,
                started_at=started_at,
            )
        except (TimeoutError, asyncio.TimeoutError):
            await self.interrupt(prompt_id)
            raise TimeoutError(f"ComfyUI output wait timed out after {timeout_seconds}s: {prompt_id}")
        if progress_callback:
            progress_callback(
                {
                    "status": "completed",
                    "progress": 100,
                    "message": "Generation completed.",
                    "elapsed_seconds": round(time.time() - started_at, 1),
                    "prompt_id": prompt_id,
                }
            )
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
        sampling_started_at: float | None = None
        async with websockets.connect(ws_url, max_size=None, ping_interval=None, ping_timeout=None) as websocket:
            while time.time() < deadline:
                try:
                    message = await asyncio.wait_for(websocket.recv(), timeout=min(5, max(1, deadline - time.time())))
                except asyncio.TimeoutError:
                    continue
                if isinstance(message, bytes):
                    continue
                event = json.loads(message)
                event_type = event.get("type")
                data = event.get("data") or {}
                event_prompt = data.get("prompt_id")
                if event_prompt and str(event_prompt) != str(prompt_id):
                    continue

                if event_type == "progress":
                    sampling_started_at = sampling_started_at or time.time()
                    value = float(data.get("value") or 0)
                    maximum = max(1.0, float(data.get("max") or 1))
                    elapsed = max(0.001, time.time() - sampling_started_at)
                    steps_per_second = value / elapsed if value > 0 else 0.0
                    eta_seconds = ((maximum - value) / steps_per_second) if steps_per_second > 0 else None
                    if progress_callback:
                        progress_callback(
                            {
                                "status": "running",
                                "progress": min(99, max(5, round((value / maximum) * 100))),
                                "current_step": int(value),
                                "total_steps": int(maximum),
                                "node": str(data.get("node") or ""),
                                "message": f"Step {int(value)}/{int(maximum)}",
                                "steps_per_second": round(steps_per_second, 3) if steps_per_second else None,
                                "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                                "elapsed_seconds": round(elapsed, 1),
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
                                "progress": 12,
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
        raise TimeoutError(f"ComfyUI progress timed out after {timeout_seconds}s: {prompt_id}")

    def _output_record_from_path(self, path: Path) -> dict[str, Any] | None:
        try:
            output_root = self.settings.output_dir.resolve()
            resolved = path.resolve()
            relative = resolved.relative_to(output_root)
        except (OSError, ValueError):
            return None
        suffix = path.suffix.lower()
        if suffix in {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz", ".3mf", ".dae", ".spz", ".ksplat"}:
            media_kind = "3d"
        elif suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
            media_kind = "video"
        elif suffix in {".png", ".jpg", ".jpeg", ".webp"}:
            media_kind = "image"
        else:
            return None
        safe_relative_path = relative.as_posix()
        return {
            "kind": media_kind,
            "filename": path.name,
            "subfolder": relative.parent.as_posix() if str(relative.parent) != "." else "",
            "type": "output",
            "path": safe_relative_path,
            "url": f"/outputs/{quote(safe_relative_path, safe='/')}",
            "detected_from_filesystem": True,
        }

    def _recent_output_files(self, started_at: float) -> list[dict[str, Any]]:
        output_root = self.settings.output_dir
        if not output_root.exists():
            return []
        suffixes = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz", ".3mf", ".dae", ".spz", ".ksplat", ".mp4", ".webm", ".mkv", ".mov", ".avi", ".png", ".jpg", ".jpeg", ".webp"}
        records: list[dict[str, Any]] = []
        for path in output_root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in suffixes:
                continue
            try:
                if path.stat().st_mtime < started_at - 2:
                    continue
            except OSError:
                continue
            record = self._output_record_from_path(path)
            if record:
                records.append(record)
        records.sort(key=lambda item: str(item.get("path") or ""))
        return records

    async def history(self, prompt_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(f"{self.base_url}/history/{prompt_id}")
            response.raise_for_status()
            return response.json()

    async def wait_for_outputs(
        self,
        prompt_id: str,
        timeout_seconds: int = 3600,
        progress_callback: Any | None = None,
        started_at: float | None = None,
    ) -> list[dict[str, Any]]:
        deadline = time.time() + timeout_seconds
        completed_since: float | None = None
        started_at = started_at or time.time()
        last_emit = 0.0
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
                    fallback_outputs = self._recent_output_files(started_at)
                    if fallback_outputs:
                        return fallback_outputs
                    completed_since = completed_since or time.time()
                    if progress_callback:
                        progress_callback(
                            {
                                "status": "polling",
                                "progress": 98,
                                "message": "Finalizing outputs",
                                "elapsed_seconds": round(time.time() - started_at, 1),
                                "prompt_id": prompt_id,
                            }
                        )
                    if time.time() - completed_since >= 30:
                        return []
            now = time.time()
            if progress_callback and now - last_emit >= 1.5:
                elapsed = max(0.0, now - started_at)
                estimated = min(95, int(12 + (83 * (elapsed / (elapsed + 90)))))
                eta_seconds = max(0.0, (elapsed / max(1, estimated - 12)) * (95 - estimated)) if estimated > 12 else None
                progress_per_second = max(0.0, (estimated - 12) / elapsed) if elapsed > 0 else 0.0
                progress_callback(
                    {
                        "status": "polling",
                        "progress": estimated,
                        "message": "Syncing ComfyUI history",
                        "elapsed_seconds": round(elapsed, 1),
                        "eta_seconds": round(eta_seconds, 1) if eta_seconds is not None else None,
                        "progress_per_second": round(progress_per_second, 3) if progress_per_second else None,
                        "prompt_id": prompt_id,
                    }
                )
                last_emit = now
            await asyncio.sleep(1)
        raise TimeoutError(f"ComfyUI job timed out: {prompt_id}")


def extract_outputs(history_item: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    model_suffixes = {".glb", ".gltf", ".obj", ".fbx", ".stl", ".ply", ".usdz", ".3mf", ".dae", ".spz", ".ksplat"}
    for node_output in history_item.get("outputs", {}).values():
        for key in ["images", "videos", "gifs", "meshes", "models", "model_files", "3d", "files"]:
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
                if suffix in model_suffixes:
                    media_kind = "3d"
                elif suffix in {".mp4", ".webm", ".mkv", ".mov", ".avi"}:
                    media_kind = "video"
                else:
                    media_kind = key[:-1]
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
