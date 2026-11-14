"""Bounded local video posters for Nexus-BTA Gallery.

Opt-in *on first thumbnail request*, never during indexing. No network URLs,
shells, model/GPU runtimes, or changes to original files. Uses an installed
FFmpeg executable with a strict deadline, file-only input protocols and
one decoded frame. Failed decodes leave no cached poster.
"""
from __future__ import annotations

import hashlib
import io
import os
import shutil
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from .gallery_thumbnail import (
    MAX_CACHE_BYTES, MAX_EDGE, safe_video_path,
    default_gallery_db_path,
)

MAX_VIDEO_INPUT_BYTES = 2 * 1024 * 1024 * 1024
MAX_DECODED_PNG_BYTES = 16 * 1024 * 1024
MAX_VIDEO_DECODE_SECONDS = 12.0


def locate_ffmpeg() -> str | None:
    """Only an installed local program, never a network download."""
    installed = shutil.which("ffmpeg")
    if installed:
        return installed
    try:
        import imageio_ffmpeg  # optional runtime dependency, includes Windows executable
        candidate = Path(imageio_ffmpeg.get_ffmpeg_exe())
        return str(candidate) if candidate.is_file() else None
    except (ImportError, OSError, RuntimeError):
        return None


def frame_command(ffmpeg_executable: str, source: Path, output_png: Path, size: int) -> list[str]:
    """Keep media-decoder command auditable and shell-free."""
    if not 64 <= size <= MAX_EDGE:
        raise ValueError("Invalid video poster size.")
    return [
        ffmpeg_executable, "-hide_banner", "-nostdin", "-loglevel", "error",
        "-protocol_whitelist", "file,pipe",
        "-probesize", "4194304", "-analyzeduration", "3000000",
        "-ss", "0", "-i", str(source),
        "-map", "0:v:0", "-frames:v", "1", "-an", "-sn", "-dn",
        "-vf", f"scale={size}:{size}:force_original_aspect_ratio=decrease:force_divisible_by=2",
        "-threads", "1", "-y", "-f", "image2", "-vcodec", "png",
        str(output_png),
    ]


class GalleryVideoPosterCache:
    def __init__(
        self,
        root: Path,
        *,
        cache_root: Path | None = None,
        executable: str | None = None,
        runner: Callable[[list[str]], None] | None = None,
    ):
        self.root = root.resolve()
        namespace = hashlib.sha256(os.fsencode(str(self.root))).hexdigest()[:20]
        default_cache = default_gallery_db_path(self.root).parent / "gallery-posters" / namespace
        self.cache_root = (cache_root or default_cache).resolve()
        if self.cache_root.is_relative_to(self.root):
            raise ValueError("Video poster cache cannot be public.")
        self.executable = executable if executable is not None else locate_ffmpeg()
        self._runner = runner
        self._lock = threading.RLock()
        self._last_trim = 0.0

    @property
    def available(self) -> bool:
        return bool(self.executable)

    def _run_decoder(self, command: list[str]) -> None:
        if self._runner is not None:
            self._runner(command)
            return
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=MAX_VIDEO_DECODE_SECONDS,
            check=False,
            shell=False,
        )
        if result.returncode != 0:
            raise ValueError("Video frame decoder returned an error.")

    def _trim_cache(self) -> None:
        files: list[tuple[float, int, Path]] = []
        total = 0
        for path in self.cache_root.glob("*.webp"):
            try:
                if path.is_symlink():
                    path.unlink()
                    continue
                stat = path.stat()
            except OSError:
                continue
            total += stat.st_size
            files.append((stat.st_mtime, stat.st_size, path))
        if total <= MAX_CACHE_BYTES:
            return
        for _, size, path in sorted(files):
            try:
                path.unlink()
                total -= size
            except OSError:
                pass
            if total <= MAX_CACHE_BYTES:
                break

    def poster(self, relative_path: str, size: int = 320) -> Path:
        if not self.available:
            raise ValueError("Video thumbnail decoder unavailable.")
        if not 64 <= size <= MAX_EDGE:
            raise ValueError("Invalid video thumbnail dimensions.")
        source = safe_video_path(self.root, relative_path)
        stat = source.stat()
        if stat.st_size > MAX_VIDEO_INPUT_BYTES:
            raise ValueError("Video exceeds thumbnail input budget.")
        key = hashlib.sha256(
            f"video-v1\0{relative_path}\0{stat.st_size}\0{stat.st_mtime_ns}\0{size}".encode("utf8")
        ).hexdigest()
        output = self.cache_root / (key + ".webp")
        with self._lock:
            if output.is_file() and not output.is_symlink():
                return output
            self.cache_root.mkdir(parents=True, exist_ok=True)
            fd, scratch = tempfile.mkstemp(prefix=".video-frame-", suffix=".png", dir=self.cache_root)
            os.close(fd)
            fd_out, scratch_out = tempfile.mkstemp(prefix=".video-thumb-", suffix=".webp", dir=self.cache_root)
            os.close(fd_out)
            try:
                try:
                    self._run_decoder(frame_command(self.executable or "", source, Path(scratch), size))
                except (subprocess.TimeoutExpired, OSError, ValueError) as exc:
                    raise ValueError("Video poster could not be generated.") from exc
                if os.path.getsize(scratch) > MAX_DECODED_PNG_BYTES:
                    raise ValueError("Decoded frame exceeds memory/disk budget.")
                try:
                    with Image.open(scratch) as frame:
                        if frame.width * frame.height > MAX_EDGE * MAX_EDGE * 4:
                            raise ValueError("Decoded frame dimensions exceed budget.")
                        frame.thumbnail((size, size), Image.Resampling.LANCZOS)
                        frame.convert("RGB").save(scratch_out, format="WEBP", quality=70, method=3)
                except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
                    raise ValueError("Invalid decoded video frame.") from exc
                os.replace(scratch_out, output)
                now = time.monotonic()
                if now - self._last_trim > 30:
                    self._trim_cache()
                    self._last_trim = now
                return output
            finally:
                for path in (scratch, scratch_out):
                    try:
                        Path(path).unlink(missing_ok=True)
                    except OSError:
                        pass
