param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$ComfyCoreSource = "C:\ComfyUI\resources\ComfyUI",
    [string]$PythonEnvSource = "C:\ComfyUpdate\.venv",
    [string]$CustomNodesSource = "C:\ComfyUpdate\custom_nodes",
    [string]$WorkflowsSource = "C:\Users\jpzin\OneDrive\Documentos\Comfy work",
    [switch]$CopyPythonEnv,
    [switch]$ImportCustomNodes,
    [switch]$ImportWorkflows,
    [switch]$ImportModels,
    [string]$ModelsSource = "C:\ComfyUpdate\models",
    [switch]$LinkModels,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Resolve-AbsolutePath([string]$PathValue) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
}

function Copy-Directory([string]$Source, [string]$Target) {
    if (!(Test-Path -LiteralPath $Source)) {
        throw "Source not found: $Source"
    }
    if ((Test-Path -LiteralPath $Target) -and $Force) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    if (!(Test-Path -LiteralPath $Target)) {
        New-Item -ItemType Directory -Path $Target | Out-Null
    }
    robocopy $Source $Target /E /NFL /NDL /NJH /NJS /NP | Out-Null
    if ($LASTEXITCODE -gt 7) {
        throw "robocopy failed from $Source to $Target with exit code $LASTEXITCODE"
    }
}

function Link-Directory([string]$Source, [string]$Target) {
    if (!(Test-Path -LiteralPath $Source)) {
        throw "Source not found: $Source"
    }
    if ((Test-Path -LiteralPath $Target) -and $Force) {
        Remove-Item -LiteralPath $Target -Recurse -Force
    }
    if (!(Test-Path -LiteralPath $Target)) {
        New-Item -ItemType Junction -Path $Target -Target $Source | Out-Null
    }
}

$root = Resolve-AbsolutePath $ProjectRoot
$ltxDirectorDeps = Join-Path $root "scripts\install_ltx_director_deps.ps1"
if (!(Test-Path -LiteralPath $root)) {
    New-Item -ItemType Directory -Path $root | Out-Null
}

$requiredDirs = @(
    "runtime",
    "models",
    "custom_nodes",
    "workflows\comfyui",
    "input",
    "output",
    "temp",
    "user",
    "config"
)

foreach ($dir in $requiredDirs) {
    New-Item -ItemType Directory -Path (Join-Path $root $dir) -Force | Out-Null
}

$modelDirs = @(
    "checkpoints",
    "diffusion_models",
    "unet",
    "loras",
    "vae",
    "text_encoders",
    "clip",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "embeddings",
    "animatediff_models",
    "animatediff_motion_lora",
    "frame_interpolation",
    "style_models",
    "diffusers"
)

foreach ($dir in $modelDirs) {
    New-Item -ItemType Directory -Path (Join-Path $root "models\$dir") -Force | Out-Null
}

$checkpointPresetDirs = @(
    "sd15",
    "sdxl",
    "flux",
    "qwen",
    "lumina",
    "wan",
    "ltx",
    "anima"
)

foreach ($dir in $checkpointPresetDirs) {
    New-Item -ItemType Directory -Path (Join-Path $root "models\checkpoints\$dir") -Force | Out-Null
}

Copy-Directory $ComfyCoreSource (Join-Path $root "runtime\ComfyUI")

if ($CopyPythonEnv) {
    Copy-Directory $PythonEnvSource (Join-Path $root "runtime\.venv")
    $runtimePython = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $comfyRequirements = Join-Path $root "runtime\ComfyUI\requirements.txt"
    if (Test-Path -LiteralPath $runtimePython) {
        if (Test-Path -LiteralPath $comfyRequirements) {
            & $runtimePython -m pip install -r $comfyRequirements
        }
        & $runtimePython -m pip install "uvicorn[standard]>=0.30" "comfyui-frontend-package>=1.43" python-multipart soundfile opencv-contrib-python
    }
}

if ($ImportCustomNodes) {
    Copy-Directory $CustomNodesSource (Join-Path $root "custom_nodes")
}

if ($ImportWorkflows) {
    Copy-Directory $WorkflowsSource (Join-Path $root "workflows\comfyui")
}

if ($ImportModels) {
    if ($LinkModels) {
        foreach ($dir in Get-ChildItem -LiteralPath $ModelsSource -Directory) {
            Link-Directory $dir.FullName (Join-Path $root "models\$($dir.Name)")
        }
    } else {
        Copy-Directory $ModelsSource (Join-Path $root "models")
    }
}

if (Test-Path -LiteralPath $ltxDirectorDeps) {
    $runtimePython = Join-Path $root "runtime\.venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $runtimePython)) {
        $runtimePython = "python"
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $ltxDirectorDeps -ProjectRoot $root -RuntimePython $runtimePython
}

Write-Host "Nexus runtime bootstrap completed at $root"
