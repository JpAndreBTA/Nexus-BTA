param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$RuntimePython = "",
    [string]$ModelsDir = "",
    [string]$CustomNodesDir = "",
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
$customNodesDir = if (![string]::IsNullOrWhiteSpace($CustomNodesDir)) {
    Get-AbsolutePath $CustomNodesDir
} elseif (![string]::IsNullOrWhiteSpace($env:NEXUS_CUSTOM_NODES_DIR)) {
    Get-AbsolutePath $env:NEXUS_CUSTOM_NODES_DIR
} else {
    Join-Path $root "custom_nodes"
}
$settingsPath = Join-Path $root "config\nexus_settings.json"
$modelsDir = if (![string]::IsNullOrWhiteSpace($ModelsDir)) {
    Get-AbsolutePath $ModelsDir
} elseif (![string]::IsNullOrWhiteSpace($env:NEXUS_MODELS_DIR)) {
    Get-AbsolutePath $env:NEXUS_MODELS_DIR
} else {
    Join-Path $root "models"
}
$diffusionModelsDir = Join-Path $modelsDir "diffusion_models"
$melDir = Join-Path $diffusionModelsDir "MelRoFormer"
$melModel = Join-Path $melDir "MelBandRoformer_fp16.safetensors"
$ltxTransitionDir = Join-Path $modelsDir "loras\ltx_transition"
$ltxTransitionLora = Join-Path $ltxTransitionDir "ltx2.3-transition.safetensors"
$ltxIcDir = Join-Path $modelsDir "loras\ltx_ic"
$ltxUnionIcLora = Join-Path $ltxIcDir "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
$ltxDetailerLora = Join-Path $ltxIcDir "ltx-2-19b-ic-lora-detailer.safetensors"
$ltxCameramanLora = Join-Path $ltxIcDir "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"
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

function Find-NexusLoraFile([string[]]$Names, [long]$MinBytes) {
    foreach ($rootPath in Get-NexusModelRootsForCategory "loras") {
        foreach ($folder in @("loras\ltx_ic", "loras\ltx_transition", "loras\ltx", "loras", "ltx_ic", "ltx_transition", "ltx")) {
            foreach ($name in $Names) {
                $candidate = Join-Path (Join-Path $rootPath $folder) $name
                if (Test-NexusModelFile $candidate $MinBytes) { return $candidate }
            }
        }
    }
    return ""
}

function Test-NexusModelFile([string]$PathValue, [long]$MinBytes) {
    if (!(Test-Path -LiteralPath $PathValue)) { return $false }
    return (Get-Item -LiteralPath $PathValue).Length -ge $MinBytes
}

function Skip-NexusModelDownload([string]$Label, [string]$PathValue) {
    if ($allowModelDownloads) { return $false }
    Write-NexusWarn "$Label is missing at $PathValue. Skipping download because startup model downloads were not approved."
    return $true
}

if (!$RuntimePython) {
    $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
}

New-Item -ItemType Directory -Force -Path $customNodesDir, $diffusionModelsDir, $melDir, $ltxTransitionDir, $ltxIcDir | Out-Null

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
        Name = "rgthree-comfy"
        Url = "https://github.com/rgthree/rgthree-comfy.git"
        Path = Join-Path $customNodesDir "rgthree-comfy"
    },
    @{
        Name = "comfyui-int-and-float"
        Url = "https://github.com/danTheMonk/comfyui-int-and-float.git"
        Path = Join-Path $customNodesDir "comfyui-int-and-float"
    },
    @{
        Name = "RES4LYF"
        Url = "https://github.com/ClownsharkBatwing/RES4LYF.git"
        Path = Join-Path $customNodesDir "RES4LYF"
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
    },
    @{
        Name = "comfyui_controlnet_aux"
        Url = "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
        Path = Join-Path $customNodesDir "comfyui_controlnet_aux"
    },
    @{
        Name = "ComfyUI-Video-Depth-Anything"
        Url = "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
        Path = Join-Path $customNodesDir "ComfyUI-Video-Depth-Anything"
    }
)

foreach ($repo in $repos) {
    Invoke-NexusStep -Label "Installing $($repo.Name)" -Step {
        if ($repo.SkipClone -and !(Test-Path -LiteralPath $repo.Path)) {
            Write-NexusWarn "$($repo.Name) is not present; install it through ComfyUI Manager or place it under $customNodesDir if the workflow requires it."
            return
        }
        if (!(Test-Path -LiteralPath $repo.Path)) {
            Write-NexusLine "Installing $($repo.Name)..." "Info"
            git clone --depth 1 $repo.Url $repo.Path
        } else {
            Write-NexusLine "$($repo.Name) is present." "Ok"
        }

        if ($repo.Commit -and (Test-Path -LiteralPath (Join-Path $repo.Path ".git"))) {
            Push-Location $repo.Path
            try {
                $localDirectorDirty = $repo.Name -eq "WhatDreamsCost-ComfyUI" -and [string]::IsNullOrWhiteSpace((git status --porcelain -- "ltx_director.py")) -eq $false
                if ($localDirectorDirty) {
                    Write-NexusLine "$($repo.Name) local LTX Director hotfixes preserved; upstream checkout skipped." "Ok"
                    return
                }
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
            Invoke-NexusPipInstallIfNeeded $repo.Name @("-r", $requirements)
        }
    }
}

Invoke-NexusStep -Label "Downloading MelBandRoFormer model" -Step {
    if (Test-NexusModelFile $melModel 100MB) {
        Write-Host "[NEXUS BTA] MelBandRoFormer model already present."
        return
    }
    if (Skip-NexusModelDownload "MelBandRoFormer model" $melModel) {
        return
    }
    if (Test-Path -LiteralPath $melModel) {
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
    $existing = Find-NexusLoraFile @("ltx2.3-transition.safetensors") 10MB
    if ($existing) {
        Write-NexusLine "LTX 2.3 Transition LoRA already present: $existing" "Ok"
        return
    }
    if (Skip-NexusModelDownload "LTX 2.3 Transition LoRA" $ltxTransitionLora) {
        return
    }
    if (Test-Path -LiteralPath $ltxTransitionLora) {
        Remove-Item -LiteralPath $ltxTransitionLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx2.3-transition.safetensors..." "Info"
    $url = "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors?download=true"
    $partial = "$ltxTransitionLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxTransitionLora -Force
}

Invoke-NexusStep -Label "Downloading LTX 2.3 IC-LoRA Union Control" -Step {
    $existing = Find-NexusLoraFile @("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors") 100MB
    if ($existing) {
        Write-NexusLine "LTX 2.3 IC-LoRA Union Control already present: $existing" "Ok"
        return
    }
    if (Test-Path -LiteralPath $ltxUnionIcLora) {
        Remove-Item -LiteralPath $ltxUnionIcLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors..." "Info"
    $url = "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors?download=true"
    $partial = "$ltxUnionIcLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxUnionIcLora -Force
}

Invoke-NexusStep -Label "Downloading LTX IC-LoRA Detailer" -Step {
    $existing = Find-NexusLoraFile @("ltx-2-19b-ic-lora-detailer.safetensors") 100MB
    if ($existing) {
        Write-NexusLine "LTX IC-LoRA Detailer already present: $existing" "Ok"
        return
    }
    if (Test-Path -LiteralPath $ltxDetailerLora) {
        Remove-Item -LiteralPath $ltxDetailerLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx-2-19b-ic-lora-detailer.safetensors..." "Info"
    $url = "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors"
    $partial = "$ltxDetailerLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxDetailerLora -Force
}

Invoke-NexusStep -Label "Downloading LTX 2.3 IC-LoRA Cameraman" -Step {
    $existing = Find-NexusLoraFile @("LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors") 100MB
    if ($existing) {
        Write-NexusLine "LTX 2.3 IC-LoRA Cameraman already present: $existing" "Ok"
        return
    }
    if (Test-Path -LiteralPath $ltxCameramanLora) {
        Remove-Item -LiteralPath $ltxCameramanLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors..." "Info"
    $url = "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1/resolve/main/LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"
    $partial = "$ltxCameramanLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxCameramanLora -Force
}

Write-NexusLine "LTX Director requirements satisfied." "Ok"
