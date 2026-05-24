param(
    [string]$ProjectRoot = "D:\NexusBTA"
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$python = Join-Path $root "runtime\.venv\Scripts\python.exe"
$bootstrap = Join-Path $root "scripts\bootstrap_nexus_runtime.ps1"
$ltxDirectorDeps = Join-Path $root "scripts\install_ltx_director_deps.ps1"

Write-Host "[NEXUS BTA] Checking repository updates..."
if (Test-Path -LiteralPath (Join-Path $root ".git")) {
    git -C $root fetch --all --prune
    git -C $root pull --ff-only
} else {
    Write-Host "[NEXUS BTA] No .git directory found; skipping git pull."
}

if (!(Test-Path -LiteralPath (Join-Path $root "runtime\ComfyUI\main.py"))) {
    Write-Host "[NEXUS BTA] Embedded ComfyUI runtime missing; bootstrapping from configured local source..."
    & $bootstrap -ProjectRoot $root -CopyPythonEnv
}

if (Test-Path -LiteralPath $python) {
    Write-Host "[NEXUS BTA] Updating Python backend dependencies..."
    & $python -m pip install --upgrade pip
    & $python -m pip install "uvicorn[standard]>=0.30" fastapi pydantic python-multipart httpx websockets pillow soundfile opencv-contrib-python

    $comfyRequirements = Join-Path $root "runtime\ComfyUI\requirements.txt"
    if (Test-Path -LiteralPath $comfyRequirements) {
        Write-Host "[NEXUS BTA] Checking embedded ComfyUI requirements..."
        & $python -m pip install -r $comfyRequirements
    }

    if (Test-Path -LiteralPath $ltxDirectorDeps) {
        Write-Host "[NEXUS BTA] Checking LTX Director dependencies..."
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ltxDirectorDeps -ProjectRoot $root -RuntimePython $python
    }
} else {
    Write-Host "[NEXUS BTA] Runtime Python not found; run run.bat after bootstrap."
}

Write-Host "[NEXUS BTA] Ensuring model and workflow folders..."
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

Write-Host "[NEXUS BTA] Dependency check complete."
