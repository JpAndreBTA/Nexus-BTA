param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$RuntimePython = "",
    [string]$ModelsDir = "",
    [switch]$IncludeWanModels,
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

function Write-NexusWarn([string]$Message) {
    if (Get-Command Write-NexusLine -ErrorAction SilentlyContinue) {
        Write-NexusLine $Message "Warn"
    } else {
        Write-Warning "[NEXUS BTA] $Message"
    }
}

function Invoke-NexusStep([scriptblock]$Step, [string]$Label) {
    try {
        & $Step
    } catch {
        if ($Strict) {
            throw
        }
        Write-NexusWarn "$Label failed: $($_.Exception.Message)"
    }
}

function Invoke-NexusPipInstallIfNeeded([string]$Label, [string[]]$PipArgs) {
    $dryArgs = @("-m", "pip", "install", "--dry-run", "--no-input", "--disable-pip-version-check", "-q") + $PipArgs
    $dryOutput = & $RuntimePython @dryArgs 2>&1
    $dryText = ($dryOutput | Out-String).Trim()
    if ($LASTEXITCODE -eq 0 -and [string]::IsNullOrWhiteSpace($dryText)) {
        Write-NexusLine "$Label requirements already satisfied." "Ok"
        return
    }

    & $RuntimePython -m pip install --disable-pip-version-check -q @PipArgs
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed with exit code $LASTEXITCODE"
    }
    Write-NexusLine "$Label requirements satisfied." "Ok"
}

function Get-AbsolutePath([string]$PathValue) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
}

$root = Get-AbsolutePath $ProjectRoot
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} elseif (!(Get-Command Write-NexusLine -ErrorAction SilentlyContinue)) {
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
}

if (!$RuntimePython) {
    $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
}

$settingsPath = Join-Path $root "config\nexus_settings.json"
$modelsDir = if (![string]::IsNullOrWhiteSpace($ModelsDir)) {
    Get-AbsolutePath $ModelsDir
} elseif (![string]::IsNullOrWhiteSpace($env:NEXUS_MODELS_DIR)) {
    Get-AbsolutePath $env:NEXUS_MODELS_DIR
} else {
    Join-Path $root "models"
}
foreach ($dir in @("diffusion_models", "text_encoders", "vae", "clip_vision", "loras")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $modelsDir $dir) | Out-Null
}
$allowModelDownloads = [string]::IsNullOrWhiteSpace($env:NEXUS_ALLOW_MODEL_DOWNLOADS) -eq $false -and $env:NEXUS_ALLOW_MODEL_DOWNLOADS -match '^(1|true|yes|y)$'

function Get-NexusModelRootsForCategory([string]$Category) {
    $roots = @($modelsDir)
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            foreach ($item in @($settingsJson.model_sources.$Category)) {
                if (![string]::IsNullOrWhiteSpace([string]$item)) { $roots += [string]$item }
            }
            foreach ($item in @($settingsJson.reference_model_sources)) {
                if (![string]::IsNullOrWhiteSpace([string]$item)) { $roots += [string]$item }
            }
        } catch {
            Write-NexusWarn "Could not read model source settings: $($_.Exception.Message)"
        }
    }
    return @($roots | Where-Object { ![string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
}

function Test-NexusWanAssetPresent($Asset) {
    $minBytes = [int64]($Asset.MinBytes)
    if ((Test-Path -LiteralPath $Asset.Target) -and (Get-Item -LiteralPath $Asset.Target).Length -ge $minBytes) {
        return $true
    }
    foreach ($alias in @($Asset.Aliases)) {
        if ($alias -and (Test-Path -LiteralPath $alias) -and (Get-Item -LiteralPath $alias).Length -ge $minBytes) {
            return $true
        }
    }
    foreach ($rootPath in Get-NexusModelRootsForCategory ([string]$Asset.Category)) {
        $candidate = Join-Path (Join-Path $rootPath ([string]$Asset.Category)) (Split-Path $Asset.Target -Leaf)
        if ((Test-Path -LiteralPath $candidate) -and (Get-Item -LiteralPath $candidate).Length -ge $minBytes) {
            return $true
        }
    }
    return $false
}

$optionalAssets = @(
    @{
        Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
        File = "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        Target = Join-Path $modelsDir "text_encoders\umt5_xxl_fp8_e4m3fn_scaled.safetensors"
        Category = "text_encoders"
        MinBytes = 1GB
    },
    @{
        Repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        File = "split_files/vae/wan_2.1_vae.safetensors"
        Target = Join-Path $modelsDir "vae\wan_2.1_vae.safetensors"
        Category = "vae"
        MinBytes = 100MB
    },
    @{
        Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
        File = "split_files/vae/wan2.2_vae.safetensors"
        Target = Join-Path $modelsDir "vae\wan2.2_vae.safetensors"
        Aliases = @((Join-Path $modelsDir "vae\wan22-vae.safetensors"))
        Category = "vae"
        MinBytes = 100MB
    }
)

$mandatoryAssets = @(
    @{
        Repo = "Comfy-Org/Wan_2.1_ComfyUI_repackaged"
        File = "split_files/clip_vision/clip_vision_h.safetensors"
        Target = Join-Path $modelsDir "clip_vision\clip_vision_h.safetensors"
        Category = "clip_vision"
        MinBytes = 500MB
    }
)

if ($IncludeWanModels) {
    $optionalAssets += @(
        @{
            Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
            File = "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            Target = Join-Path $modelsDir "diffusion_models\wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors"
            Category = "diffusion_models"
            MinBytes = 1GB
        },
        @{
            Repo = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
            File = "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            Target = Join-Path $modelsDir "diffusion_models\wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors"
            Category = "diffusion_models"
            MinBytes = 1GB
        }
    )
}

$assets = @($mandatoryAssets | Where-Object { -not (Test-NexusWanAssetPresent $_) })
if (!$allowModelDownloads) {
    $missing = @($optionalAssets | Where-Object { -not (Test-NexusWanAssetPresent $_) })
    if ($missing.Count -gt 0) {
        Write-NexusWarn ("Skipping Wan 2.2 model downloads because startup model downloads were not approved: {0}" -f (($missing | ForEach-Object { Split-Path $_.Target -Leaf }) -join ", "))
    }
} else {
    $assets += @($optionalAssets | Where-Object { -not (Test-NexusWanAssetPresent $_) })
}

if ($assets.Count -gt 0) {
    Invoke-NexusStep -Label "Installing Hugging Face downloader" -Step {
        Invoke-NexusPipInstallIfNeeded "Hugging Face downloader" @("huggingface-hub>=0.24")
    }
} else {
    Write-NexusLine "Wan 2.2 model downloads skipped or already satisfied." "Info"
}

$assetJson = $assets | ConvertTo-Json -Depth 5 -Compress
$downloadScript = @'
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

from huggingface_hub import hf_hub_download

assets = json.loads(os.environ["NEXUS_WAN22_ASSETS"])

for asset in assets:
    target = Path(asset["Target"])
    min_bytes = int(asset.get("MinBytes") or 1)
    if target.exists() and target.stat().st_size >= min_bytes:
        print(f"[NEXUS BTA] Wan asset present: {target.name}")
        continue
    alias_hit = None
    for alias in asset.get("Aliases") or []:
        alias_path = Path(alias)
        if alias_path.exists() and alias_path.stat().st_size >= min_bytes:
            alias_hit = alias_path
            break
    if alias_hit:
        print(f"[NEXUS BTA] Wan asset present: {alias_hit.name}")
        continue

    target.parent.mkdir(parents=True, exist_ok=True)
    tmpdir = Path(tempfile.mkdtemp(prefix="nexus-wan22-"))
    try:
        print(f"[NEXUS BTA] Downloading {asset['File']}...")
        downloaded = Path(hf_hub_download(
            repo_id=asset["Repo"],
            filename=asset["File"],
            local_dir=tmpdir,
            local_dir_use_symlinks=False,
            resume_download=True,
        ))
        partial = target.with_suffix(target.suffix + ".part")
        if partial.exists():
            partial.unlink()
        shutil.copy2(downloaded, partial)
        partial.replace(target)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
'@

if ($assets.Count -gt 0) {
    Invoke-NexusStep -Label "Downloading Wan 2.2 support assets" -Step {
        $env:NEXUS_WAN22_ASSETS = $assetJson
        $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) ("nexus_wan22_assets_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
        try {
            Set-Content -LiteralPath $tempScript -Value $downloadScript -Encoding UTF8
            & $RuntimePython $tempScript
            if ($LASTEXITCODE -ne 0) {
                throw "Wan 2.2 asset downloader failed with exit code $LASTEXITCODE"
            }
        } finally {
            Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
            Remove-Item Env:\NEXUS_WAN22_ASSETS -ErrorAction SilentlyContinue
        }
    }
}

Write-NexusLine "Wan 2.2 support assets satisfied." "Ok"
