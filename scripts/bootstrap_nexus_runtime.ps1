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

function Resolve-BootstrapPython {
    $candidates = @(
        @{ File = "py"; Args = @("-3.11") },
        @{ File = "py"; Args = @("-3.12") },
        @{ File = "python"; Args = @() }
    )
    foreach ($candidate in $candidates) {
        try {
            $cmd = Get-Command $candidate.File -ErrorAction Stop
            $version = & $cmd.Source @($candidate.Args + @("-c", "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")) 2>$null
            if ($LASTEXITCODE -eq 0 -and $version -match "^(3\.1[01]|3\.12)$") {
                return @{ File = $cmd.Source; Args = $candidate.Args }
            }
        } catch {
            continue
        }
    }
    throw "Python 3.10, 3.11 or 3.12 was not found. Install Python 3.11/3.12 and run run.bat again."
}

$root = Resolve-AbsolutePath $ProjectRoot
$ltxDirectorDeps = Join-Path $root "scripts\install_ltx_director_deps.ps1"
$wan22Deps = Join-Path $root "scripts\install_wan22_deps.ps1"
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

$comfyTarget = Join-Path $root "runtime\ComfyUI"
if (Test-Path -LiteralPath $ComfyCoreSource) {
    Copy-Directory $ComfyCoreSource $comfyTarget
} elseif (!(Test-Path -LiteralPath (Join-Path $comfyTarget "main.py"))) {
    git clone --depth 1 https://github.com/comfyanonymous/ComfyUI.git $comfyTarget
}

if ($CopyPythonEnv -and (Test-Path -LiteralPath $PythonEnvSource)) {
    Copy-Directory $PythonEnvSource (Join-Path $root "runtime\.venv")
}

if ($CopyPythonEnv -or !(Test-Path -LiteralPath (Join-Path $root "runtime\.venv\Scripts\python.exe"))) {
    $runtimePython = Join-Path $root "runtime\.venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $runtimePython)) {
        $bootstrapPython = Resolve-BootstrapPython
        & $bootstrapPython.File @($bootstrapPython.Args + @("-m", "venv", (Join-Path $root "runtime\.venv")))
    }
    $comfyRequirements = Join-Path $root "runtime\ComfyUI\requirements.txt"
    $nexusRequirements = Join-Path $root "requirements.txt"
    if (Test-Path -LiteralPath $runtimePython) {
        & $runtimePython -m pip install --upgrade pip wheel setuptools
        if (Test-Path -LiteralPath $comfyRequirements) {
            & $runtimePython -m pip install -r $comfyRequirements
        }
        if (Test-Path -LiteralPath $nexusRequirements) {
            & $runtimePython -m pip install -r $nexusRequirements
        } else {
            & $runtimePython -m pip install "uvicorn[standard]>=0.30" "comfyui-frontend-package>=1.43" python-multipart soundfile opencv-contrib-python
        }
    }
}

if ($ImportCustomNodes -and (Test-Path -LiteralPath $CustomNodesSource)) {
    Copy-Directory $CustomNodesSource (Join-Path $root "custom_nodes")
}

if ($ImportWorkflows -and (Test-Path -LiteralPath $WorkflowsSource)) {
    Copy-Directory $WorkflowsSource (Join-Path $root "workflows\comfyui")
}

if ($ImportModels -and (Test-Path -LiteralPath $ModelsSource)) {
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

if (Test-Path -LiteralPath $wan22Deps) {
    $runtimePython = Join-Path $root "runtime\.venv\Scripts\python.exe"
    if (!(Test-Path -LiteralPath $runtimePython)) {
        $runtimePython = "python"
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $wan22Deps -ProjectRoot $root -RuntimePython $runtimePython
}

Write-Host "Nexus runtime bootstrap completed at $root"
