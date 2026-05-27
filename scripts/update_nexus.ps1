param(
    [string]$ProjectRoot = "D:\NexusBTA"
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$python = Join-Path $root "runtime\.venv\Scripts\python.exe"
$bootstrap = Join-Path $root "scripts\bootstrap_nexus_runtime.ps1"
$ltxDirectorDeps = Join-Path $root "scripts\install_ltx_director_deps.ps1"
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"

if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLogo { Write-Host "[NEXUS BTA]" }
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
    function Write-NexusSection([string]$Title) { Write-Host ""; Write-NexusLine $Title "Step" }
    function Invoke-NexusRepositoryUpdate([string]$ProjectRoot, [switch]$Strict) {
        git -C $ProjectRoot fetch --all --prune
        git -C $ProjectRoot pull --ff-only
    }
}

Write-NexusLogo
Write-NexusSection "Updates"
Invoke-NexusRepositoryUpdate -ProjectRoot $root -Strict

if (!(Test-Path -LiteralPath (Join-Path $root "runtime\ComfyUI\main.py"))) {
    Write-NexusLine "Embedded ComfyUI is missing; preparing local runtime..." "Warn"
    & $bootstrap -ProjectRoot $root -CopyPythonEnv
}

if (Test-Path -LiteralPath $python) {
    Write-NexusSection "Requirements"
    Write-NexusLine "Backend Python..." "Info"
    & $python -m pip install -q --upgrade pip
    $nexusRequirements = Join-Path $root "requirements.txt"
    if (Test-Path -LiteralPath $nexusRequirements) {
        & $python -m pip install -q -r $nexusRequirements
    } else {
        & $python -m pip install -q "uvicorn[standard]>=0.30" fastapi pydantic python-multipart httpx websockets pillow soundfile opencv-contrib-python
    }
    Write-NexusLine "Backend Python requirements satisfied." "Ok"

    $comfyRequirements = Join-Path $root "runtime\ComfyUI\requirements.txt"
    if (Test-Path -LiteralPath $comfyRequirements) {
        Write-NexusLine "ComfyUI Python..." "Info"
        & $python -m pip install -q -r $comfyRequirements
        Write-NexusLine "ComfyUI Python requirements satisfied." "Ok"
    }

    if (Test-Path -LiteralPath $ltxDirectorDeps) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ltxDirectorDeps -ProjectRoot $root -RuntimePython $python
    }
} else {
    Write-NexusLine "Runtime Python not found; run run.bat after bootstrap." "Warn"
}

Write-NexusSection "Folders"
New-Item -ItemType Directory -Force -Path (Join-Path $root "workflows\nexus_base") | Out-Null
foreach ($dir in @(
    "models\checkpoints\sd15",
    "models\checkpoints\sdxl",
    "models\checkpoints\flux",
    "models\checkpoints\qwen",
    "models\checkpoints\lumina",
    "models\checkpoints\wan",
    "models\checkpoints\ltx",
    "models\checkpoints\anima",
    "models\loras",
    "models\vae",
    "models\text_encoders",
    "models\clip_vision",
    "models\controlnet",
    "models\upscale_models",
    "models\latent_upscale_models",
    "models\embeddings"
)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $root $dir) | Out-Null
}

Write-NexusLine "Model and workflow folders are ready." "Ok"
Write-NexusLine "Verification complete." "Ok"
