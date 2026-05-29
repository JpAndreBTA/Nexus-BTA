param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$RuntimePython = "",
    [string]$ModelsDir = "",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
    function Write-NexusWarn([string]$Message) { Write-NexusLine $Message "Warn" }
}
if (!(Get-Command Invoke-NexusStep -ErrorAction SilentlyContinue)) {
    function Invoke-NexusStep([string]$Label, [scriptblock]$Step) { Write-NexusLine $Label "Info"; & $Step }
}
if (!(Get-Command Write-NexusWarn -ErrorAction SilentlyContinue)) {
    function Write-NexusWarn([string]$Message) { Write-NexusLine $Message "Warn" }
}

if (!$RuntimePython) {
    $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
}

if ([string]::IsNullOrWhiteSpace($ModelsDir)) {
    $ModelsDir = if (![string]::IsNullOrWhiteSpace($env:NEXUS_MODELS_DIR)) { $env:NEXUS_MODELS_DIR } else { Join-Path $root "models" }
}

$modelsDir = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ModelsDir)
$dinov3Dir = Join-Path $modelsDir "facebook\dinov3-vitl16-pretrain-lvd1689m"
$dinov3Model = Join-Path $dinov3Dir "model.safetensors"
$allowModelDownloads = [string]::IsNullOrWhiteSpace($env:NEXUS_ALLOW_MODEL_DOWNLOADS) -eq $false -and $env:NEXUS_ALLOW_MODEL_DOWNLOADS -match '^(1|true|yes|y)$'

Invoke-NexusStep -Label "Installing DINOv3 Python support" -Step {
    & $RuntimePython -m pip install -q "transformers>=4.57.6,<5" "kagglehub==0.3.13"
    if ($LASTEXITCODE -ne 0) {
        throw "pip install DINOv3 support dependencies failed with exit code $LASTEXITCODE"
    }
    Write-NexusLine "DINOv3 Python support requirements satisfied." "Ok"
}

if (Test-Path -LiteralPath $dinov3Model) {
    $size = (Get-Item -LiteralPath $dinov3Model).Length
    if ($size -gt 100MB) {
        Write-NexusLine "DINOv3 model already present: $dinov3Model" "Ok"
        return
    }
}

if (!$allowModelDownloads) {
    Write-NexusWarn "DINOv3 model is missing at $dinov3Model. Skipping heavy model download because model downloads were not approved."
    return
}

$downloadScript = @'
import shutil
from pathlib import Path

import kagglehub

handle = "x1an9l1/facebook-dinov3-vitl16-pretrain-lvd1689m/transformers/default"
target = Path(r"__TARGET_DIR__")
source = Path(kagglehub.model_download(handle))
nested = source / "facebook" / "dinov3-vitl16-pretrain-lvd1689m"
if (nested / "model.safetensors").exists():
    source = nested
if not (source / "model.safetensors").exists():
    raise RuntimeError(f"Kaggle DINOv3 download did not contain model.safetensors under {source}")
target.mkdir(parents=True, exist_ok=True)
for item in source.iterdir():
    dest = target / item.name
    if item.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(item, dest)
    else:
        shutil.copy2(item, dest)
print(f"[NEXUS BTA] DINOv3 Kaggle mirror copied to {target}")
'@

Invoke-NexusStep -Label "Downloading DINOv3 from Kaggle mirror" -Step {
    $tempScript = Join-Path ([System.IO.Path]::GetTempPath()) ("nexus_dinov3_kaggle_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    try {
        $scriptText = $downloadScript.Replace("__TARGET_DIR__", $dinov3Dir.Replace("\", "\\"))
        Set-Content -LiteralPath $tempScript -Value $scriptText -Encoding UTF8
        & $RuntimePython $tempScript
        if ($LASTEXITCODE -ne 0) {
            throw "DINOv3 Kaggle downloader failed with exit code $LASTEXITCODE"
        }
    } finally {
        Remove-Item -LiteralPath $tempScript -Force -ErrorAction SilentlyContinue
    }
}

if (!(Test-Path -LiteralPath $dinov3Model) -or (Get-Item -LiteralPath $dinov3Model).Length -le 100MB) {
    throw "DINOv3 download did not create a valid model.safetensors at $dinov3Model"
}

Write-NexusLine "DINOv3 requirements satisfied." "Ok"
