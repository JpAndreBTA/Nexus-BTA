param(
    [string]$ProjectRoot = "D:\NexusBTA"
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$qwenNodes = Join-Path $root "runtime\ComfyUI\comfy_extras\nodes_qwen.py"
$ltxDirectorNode = Join-Path $root "custom_nodes\WhatDreamsCost-ComfyUI\ltx_director.py"

if (Test-Path -LiteralPath $qwenNodes) {
    $content = Get-Content -LiteralPath $qwenNodes -Raw
    if ($content -match "node_helpers\.conditioning_set_values") {
        $patched = $content -replace "(?m)^import node_helpers\r?\n", ""
        if ($patched -match "(?m)^import comfy\.model_management\r?$") {
            $patched = $patched -replace "(?m)^(import comfy\.model_management\r?\n)", "`$1import node_helpers`r`n"
        } else {
            $patched = "import node_helpers`r`n$patched"
        }
        if ($patched -ne $content) {
            Set-Content -LiteralPath $qwenNodes -Value $patched -Encoding UTF8
            Write-Host "[NEXUS BTA] Applied ComfyUI Qwen node_helpers hotfix."
        }
    }
}

if (Test-Path -LiteralPath $ltxDirectorNode) {
    $content = Get-Content -LiteralPath $ltxDirectorNode -Raw
    if ($content -notmatch "NEXUS_SEGMENT_CROP_HOTFIX") {
        $helper = @'

# NEXUS_SEGMENT_CROP_HOTFIX
def _nexus_apply_segment_crop(img, seg):
    try:
        crop = seg.get("crop") or {}
        x = max(0.0, min(1.0, float(crop.get("x", 0.0))))
        y = max(0.0, min(1.0, float(crop.get("y", 0.0))))
        w = max(0.001, min(1.0 - x, float(crop.get("w", 1.0))))
        h = max(0.001, min(1.0 - y, float(crop.get("h", 1.0))))
        if x <= 0 and y <= 0 and w >= 0.999 and h >= 0.999:
            return img
        width, height = img.size
        left = int(width * x)
        top = int(height * y)
        right = max(left + 1, int(width * min(1.0, x + w)))
        bottom = max(top + 1, int(height * min(1.0, y + h)))
        return img.crop((left, top, right, bottom))
    except Exception:
        return img
'@
        $content = $content -replace 'GuideData = io\.Custom\("GUIDE_DATA"\)\r?\n', "GuideData = io.Custom(`"GUIDE_DATA`")`r`n$helper`r`n"
        $content = $content.Replace(
            "            img = Image.open(file_path).convert(`"RGB`")`r`n            arr = np.array(img, dtype=np.float32) / 255.0",
            "            img = Image.open(file_path).convert(`"RGB`")`r`n            img = _nexus_apply_segment_crop(img, seg)`r`n            arr = np.array(img, dtype=np.float32) / 255.0"
        )
        $content = $content.Replace(
            "        img = Image.open(_io.BytesIO(img_bytes)).convert(`"RGB`")`r`n        arr = np.array(img, dtype=np.float32) / 255.0",
            "        img = Image.open(_io.BytesIO(img_bytes)).convert(`"RGB`")`r`n        img = _nexus_apply_segment_crop(img, seg)`r`n        arr = np.array(img, dtype=np.float32) / 255.0"
        )
        Set-Content -LiteralPath $ltxDirectorNode -Value $content -Encoding UTF8
        Write-Host "[NEXUS BTA] Applied WhatDreamsCost LTX Director per-segment crop hotfix."
    }
    if ($content -notmatch "NEXUS_VIDEO_SEGMENT_FRAME_HOTFIX") {
        $videoHelper = @'

# NEXUS_VIDEO_SEGMENT_FRAME_HOTFIX
def _nexus_load_video_frame_tensor(seg):
    try:
        video_ref = str(seg.get("videoFile") or "").strip()
        container = None
        if video_ref and not video_ref.startswith("data:"):
            candidates = [
                video_ref,
                os.path.join(folder_paths.get_input_directory(), video_ref),
            ]
            for candidate in candidates:
                if candidate and os.path.exists(candidate):
                    container = av.open(candidate)
                    break
        if container is None:
            b64_str = seg.get("videoB64", "") or video_ref
            if not b64_str or not str(b64_str).startswith("data:video/"):
                return None
            if "," in b64_str:
                b64_str = b64_str.split(",", 1)[1]
            container = av.open(_io.BytesIO(base64.b64decode(b64_str)))

        stream = container.streams.video[0] if len(container.streams.video) else None
        if stream is None:
            container.close()
            return None
        load_video = seg.get("loadVideo") or {}
        start_time = float(load_video.get("start_time", 0.0) or seg.get("trimStart", 0.0) or 0.0)
        if start_time > 0 and stream.time_base:
            try:
                container.seek(int(start_time / float(stream.time_base)), stream=stream, any_frame=False, backward=True)
            except Exception:
                pass
        for frame in container.decode(stream):
            img = frame.to_image().convert("RGB")
            img = _nexus_apply_segment_crop(img, seg)
            arr = np.array(img, dtype=np.float32) / 255.0
            container.close()
            return torch.from_numpy(arr).unsqueeze(0)
        container.close()
    except Exception as e:
        log.warning("[PromptRelay] Could not extract Director video guide frame: %s", e)
    return None
'@
        $content = $content -replace '(?m)^def _load_image_tensor\(seg: dict\) -> torch\.Tensor:', "$videoHelper`r`ndef _load_image_tensor(seg: dict) -> torch.Tensor:"
        $videoSegmentFilter = 'if s.get("type", "image") in {"image", "video"}' + "`r`n" + '                and (s.get("imageFile") or s.get("imageB64") or s.get("videoFile") or s.get("videoB64"))'
        $content = $content -replace 'if s\.get\("type", "image"\) == "image"\s+and \(s\.get\("imageFile"\) or s\.get\("imageB64"\)\)', $videoSegmentFilter
        $content = $content -replace '(?m)^    if seg\.get\("imageFile"\):', "    if seg.get(`"type`") == `"video`" and (seg.get(`"videoFile`") or seg.get(`"videoB64`")):`r`n        tensor = _nexus_load_video_frame_tensor(seg)`r`n        if tensor is not None:`r`n            return tensor`r`n`r`n    if seg.get(`"imageFile`"):"
        Set-Content -LiteralPath $ltxDirectorNode -Value $content -Encoding UTF8
        Write-Host "[NEXUS BTA] Applied WhatDreamsCost LTX Director video frame guide hotfix."
    }
    if ($content -notmatch "NEXUS_SEGMENT_RESIZE_METHOD_HOTFIX") {
        $resizeHelper = @'

# NEXUS_SEGMENT_RESIZE_METHOD_HOTFIX
def _nexus_segment_resize_method(seg, fallback):
    try:
        load_video = seg.get("loadVideo") or {}
        value = (
            seg.get("resizeMethod")
            or seg.get("resize_method")
            or load_video.get("resize_method")
            or fallback
        )
        value = str(value or fallback or "maintain aspect ratio").strip()
        aliases = {
            "keep proportion": "maintain aspect ratio",
            "stretch": "stretch to fit",
        }
        return aliases.get(value, value)
    except Exception:
        return fallback
'@
        if ($content -match "NEXUS_VIDEO_SEGMENT_FRAME_HOTFIX") {
            $content = $content -replace "(?m)^# NEXUS_VIDEO_SEGMENT_FRAME_HOTFIX", "$resizeHelper`r`n# NEXUS_VIDEO_SEGMENT_FRAME_HOTFIX"
        } else {
            $content = $content -replace "(?m)^def _load_image_tensor\(seg: dict\) -> torch\.Tensor:", "$resizeHelper`r`ndef _load_image_tensor(seg: dict) -> torch.Tensor:"
        }
        $content = $content.Replace(
            "tensor = _resize_image(tensor, custom_width, custom_height, resize_method, divisible_by)",
            "tensor = _resize_image(tensor, custom_width, custom_height, _nexus_segment_resize_method(seg, resize_method), divisible_by)"
        )
        Set-Content -LiteralPath $ltxDirectorNode -Value $content -Encoding UTF8
        Write-Host "[NEXUS BTA] Applied WhatDreamsCost LTX Director per-segment resize method hotfix."
    }
    if ($content -match "NEXUS_DIRECTOR_AUDIO_NORMALIZE_HOTFIX") {
        $content = $content -replace "    # NEXUS_DIRECTOR_AUDIO_NORMALIZE_HOTFIX\r?\n    peak = float\(out_waveform\.abs\(\)\.max\(\)\.item\(\)\) if out_waveform\.numel\(\) else 0\.0\r?\n    if peak > 0\.001:\r?\n        out_waveform = torch\.clamp\(out_waveform \* min\(64\.0, 0\.85 / peak\), -1\.0, 1\.0\)\r?\n", "    # NEXUS_DIRECTOR_AUDIO_ATTENUATE_ONLY`r`n"
        Set-Content -LiteralPath $ltxDirectorNode -Value $content -Encoding UTF8
        Write-Host "[NEXUS BTA] Replaced LTX Director peak-boost audio hotfix with attenuate-only handling."
    } elseif ($content -notmatch "NEXUS_DIRECTOR_AUDIO_ATTENUATE_ONLY") {
        $content = $content.Replace(
            "    return {`"waveform`": out_waveform.unsqueeze(0), `"sample_rate`": target_sr}",
            "    # NEXUS_DIRECTOR_AUDIO_ATTENUATE_ONLY`r`n    return {`"waveform`": out_waveform.unsqueeze(0), `"sample_rate`": target_sr}"
        )
        Set-Content -LiteralPath $ltxDirectorNode -Value $content -Encoding UTF8
        Write-Host "[NEXUS BTA] Applied WhatDreamsCost LTX Director attenuate-only audio guard."
    }
    if ($content -notmatch "_nexus_audio_latent_noise_mask") {
        $audioMaskHelper = @'


# NEXUS_AUDIO_LATENT_TIMELINE_MASK
def _nexus_text_requests_lipsync(seg: dict) -> bool:
    text = str(seg.get("prompt") or seg.get("text") or "").lower()
    audio_cues = (
        "custom audio", "audio custom", "musica", "música", "song", "music",
    )
    speech_cues = (
        " says ", " said ", " speak", " speaks", " speaking", " voice", "dialogue",
        "fala", "falando", "diz ", "voz", "dialogo", "diálogo", "\"", "“", "”", "'",
    )
    layer_cues = (
        "ambient", "ambiente", "room tone", "soundscape", "efeito sonoro",
        "efeitos sonoros", "sound effect", "sound effects", "people talking",
        "pessoas falando", "background voices", "crowd", "foley",
    )
    return (
        "lip sync" in text
        or "lip-sync" in text
        or "sincron" in text
        or any(cue in text for cue in layer_cues)
        or (any(cue in text for cue in audio_cues) and any(cue in f" {text} " for cue in speech_cues))
    )


def _nexus_audio_latent_noise_mask(timeline_data_str: str, frame_rate: float, latent_samples):
    time_bins = int(latent_samples.shape[-2])
    freq_bins = int(latent_samples.shape[-1])
    mask_1d = torch.ones((time_bins,), dtype=torch.float32, device=latent_samples.device)
    try:
        data = json.loads(timeline_data_str) if timeline_data_str else {}
    except Exception:
        data = {}
    audio_segments = [seg for seg in data.get("audioSegments", []) if isinstance(seg, dict)]
    text_segments = [seg for seg in data.get("segments", []) if isinstance(seg, dict) and str(seg.get("type", "text")) == "text"]
    max_frame = 1.0
    for seg in audio_segments + text_segments:
        try:
            max_frame = max(max_frame, float(seg.get("start", 0)) + float(seg.get("length", 1)))
        except Exception:
            pass
    def frame_to_bin(frame_value: float) -> int:
        if max_frame <= 0:
            return 0
        return max(0, min(time_bins, int(round((float(frame_value) / max_frame) * time_bins))))
    for seg in audio_segments:
        try:
            start = float(seg.get("start", 0))
            end = start + float(seg.get("length", 1))
        except Exception:
            continue
        mask_1d[frame_to_bin(start):max(frame_to_bin(start) + 1, frame_to_bin(end))] = 0.0
    for seg in text_segments:
        if not _nexus_text_requests_lipsync(seg):
            continue
        try:
            start = float(seg.get("start", 0))
            end = start + float(seg.get("length", 1))
        except Exception:
            continue
        start_bin = frame_to_bin(start)
        end_bin = max(start_bin + 1, frame_to_bin(end))
        mask_1d[start_bin:end_bin] = torch.maximum(
            mask_1d[start_bin:end_bin],
            torch.full_like(mask_1d[start_bin:end_bin], 0.65),
        )
    return mask_1d[:, None].repeat(1, freq_bins).reshape((1, 1, time_bins, freq_bins))
'@
        $content = $content -replace '(?m)^def _convert_to_latent_lengths\(', "$audioMaskHelper`r`ndef _convert_to_latent_lengths("
        $solidMaskPattern = '(?s)                        # 2\. Create solid mask with value 0\.0.*?                        # 3\. Set Latent Noise Mask'
        $content = [regex]::Replace(
            $content,
            $solidMaskPattern,
            "                        # 2. Keep custom audio where clips exist, but allow text-only/speech regions to generate voice or room tone.`r`n                        mask = _nexus_audio_latent_noise_mask(timeline_data, float(frame_rate), latent_samples)`r`n`r`n                        # 3. Set Latent Noise Mask"
        )
        $content = $content.Replace(
            'log.info("[PromptRelay] Generated custom audio latent with noise mask (value=0.0).")',
            'log.info("[PromptRelay] Generated custom audio latent with timeline-aware noise mask.")'
        )
        Set-Content -LiteralPath $ltxDirectorNode -Value $content -Encoding UTF8
        Write-Host "[NEXUS BTA] Applied LTX Director timeline-aware audio latent mask hotfix."
    }
}
