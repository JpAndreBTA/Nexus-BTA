param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$RuntimePython = "",
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
$customNodesDir = Join-Path $root "custom_nodes"
$modelsDir = Join-Path $root "models"
$diffusionModelsDir = Join-Path $modelsDir "diffusion_models"
$melDir = Join-Path $diffusionModelsDir "MelRoFormer"
$melModel = Join-Path $melDir "MelBandRoformer_fp16.safetensors"
$ltxTransitionDir = Join-Path $modelsDir "loras\ltx_transition"
$ltxTransitionLora = Join-Path $ltxTransitionDir "ltx2.3-transition.safetensors"

if (!$RuntimePython) {
    $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
}

New-Item -ItemType Directory -Force -Path $customNodesDir, $diffusionModelsDir, $melDir, $ltxTransitionDir | Out-Null

$repos = @(
    @{
        Name = "ComfyUI-LTXVideo"
        Url = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
        Path = Join-Path $customNodesDir "ComfyUI-LTXVideo"
    },
    @{
        Name = "ComfyUI-VideoHelperSuite"
        Url = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
        Path = Join-Path $customNodesDir "ComfyUI-VideoHelperSuite"
    },
    @{
        Name = "ComfyUI-KJNodes"
        Url = "https://github.com/kijai/ComfyUI-KJNodes.git"
        Path = Join-Path $customNodesDir "comfyui-kjnodes"
    },
    @{
        Name = "WhatDreamsCost-ComfyUI"
        Url = "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git"
        Path = Join-Path $customNodesDir "WhatDreamsCost-ComfyUI"
        Commit = "f0c8a322eaa607eb499ebd320f0d7d03f4caff80"
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
            Write-NexusLine "Installing $($repo.Name)..." "Info"
            git clone --depth 1 $repo.Url $repo.Path
        } else {
            Write-NexusLine "$($repo.Name) is present." "Ok"
        }

        if ($repo.Commit -and (Test-Path -LiteralPath (Join-Path $repo.Path ".git"))) {
            Push-Location $repo.Path
            try {
                $currentCommit = (git rev-parse HEAD).Trim()
                if ($currentCommit -ne $repo.Commit) {
                    Write-NexusLine "Pinning $($repo.Name) to $($repo.Commit.Substring(0, 7))..." "Info"
                    git fetch origin $repo.Commit --depth 1
                    git checkout --detach $repo.Commit
                }
            } finally {
                Pop-Location
            }
        }

        $requirements = Join-Path $repo.Path "requirements.txt"
        if (Test-Path -LiteralPath $requirements) {
            Write-NexusLine "$($repo.Name) Python requirements..." "Info"
            & $RuntimePython -m pip install -q -r $requirements
            Write-NexusLine "$($repo.Name) requirements satisfied." "Ok"
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
    Write-NexusLine "Downloading MelBandRoformer_fp16.safetensors..." "Info"
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $melModel -Force
}

Invoke-NexusStep -Label "Downloading LTX 2.3 Transition LoRA" -Step {
    if (Test-Path -LiteralPath $ltxTransitionLora) {
        $size = (Get-Item -LiteralPath $ltxTransitionLora).Length
        if ($size -gt 10MB) {
            Write-NexusLine "LTX 2.3 Transition LoRA already present." "Ok"
            return
        }
        Remove-Item -LiteralPath $ltxTransitionLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx2.3-transition.safetensors..." "Info"
    $url = "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors?download=true"
    $partial = "$ltxTransitionLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxTransitionLora -Force
}

Write-NexusLine "LTX Director requirements satisfied." "Ok"
