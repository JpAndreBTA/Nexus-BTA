param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$RuntimePython = "",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

function Write-NexusWarn([string]$Message) {
    Write-Warning "[NEXUS BTA] $Message"
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

function Get-AbsolutePath([string]$PathValue) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
}

$root = Get-AbsolutePath $ProjectRoot
$customNodesDir = Join-Path $root "custom_nodes"
$modelsDir = Join-Path $root "models"
$diffusionModelsDir = Join-Path $modelsDir "diffusion_models"
$melDir = Join-Path $diffusionModelsDir "MelRoFormer"
$melModel = Join-Path $melDir "MelBandRoformer_fp16.safetensors"

if (!$RuntimePython) {
    $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
}

New-Item -ItemType Directory -Force -Path $customNodesDir, $diffusionModelsDir, $melDir | Out-Null

$repos = @(
    @{
        Name = "WhatDreamsCost-ComfyUI"
        Url = "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git"
        Path = Join-Path $customNodesDir "WhatDreamsCost-ComfyUI"
    },
    @{
        Name = "ComfyUI-MelBandRoFormer"
        Url = "https://github.com/kijai/ComfyUI-MelBandRoFormer.git"
        Path = Join-Path $customNodesDir "ComfyUI-MelBandRoFormer"
    }
)

foreach ($repo in $repos) {
    Invoke-NexusStep -Label "Installing $($repo.Name)" -Step {
        if (!(Test-Path -LiteralPath $repo.Path)) {
            Write-Host "[NEXUS BTA] Cloning $($repo.Name)..."
            git clone --depth 1 $repo.Url $repo.Path
        } else {
            Write-Host "[NEXUS BTA] $($repo.Name) already present."
        }

        $requirements = Join-Path $repo.Path "requirements.txt"
        if (Test-Path -LiteralPath $requirements) {
            Write-Host "[NEXUS BTA] Checking $($repo.Name) Python requirements..."
            & $RuntimePython -m pip install -r $requirements
        }
    }
}

Invoke-NexusStep -Label "Downloading MelBandRoFormer model" -Step {
    if (Test-Path -LiteralPath $melModel) {
        $size = (Get-Item -LiteralPath $melModel).Length
        if ($size -gt 100MB) {
            Write-Host "[NEXUS BTA] MelBandRoFormer model already present."
            return
        }
        Remove-Item -LiteralPath $melModel -Force -ErrorAction SilentlyContinue
    }

    $url = "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors?download=true"
    $partial = "$melModel.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Write-Host "[NEXUS BTA] Downloading MelBandRoformer_fp16.safetensors to models\diffusion_models\MelRoFormer..."
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $melModel -Force
}

Write-Host "[NEXUS BTA] LTX Director dependency check complete."
