param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$python = Join-Path $root "runtime\.venv\Scripts\python.exe"
$bootstrap = Join-Path $root "scripts\bootstrap_nexus_runtime.ps1"
$customNodeDeps = Join-Path $root "scripts\install_comfy_custom_node_deps.ps1"
$ltxDirectorDeps = Join-Path $root "scripts\install_ltx_director_deps.ps1"
$wan22Deps = Join-Path $root "scripts\install_wan22_deps.ps1"
$minimaxH3Deps = Join-Path $root "scripts\install_minimax_h3_deps.ps1"
$minimaxH3Workflows = Join-Path $root "scripts\import_minimax_h3_workflows.ps1"
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
$settingsPath = Join-Path $root "config\nexus_settings.json"

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

function Get-NexusConfiguredModelsDir {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_MODELS_DIR)) {
        return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:NEXUS_MODELS_DIR)
    }
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            if (![string]::IsNullOrWhiteSpace([string]$settingsJson.models_dir)) {
                return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([string]$settingsJson.models_dir)
            }
        } catch {
            Write-NexusLine "Could not read configured model path; falling back to ./models." "Warn"
        }
    }
    return Join-Path $root "models"
}

$modelsDir = Get-NexusConfiguredModelsDir

function Get-NexusConfiguredComfyRoot {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_COMFY_ROOT)) {
        return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:NEXUS_COMFY_ROOT)
    }
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            if (![string]::IsNullOrWhiteSpace([string]$settingsJson.comfy_root)) {
                return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([string]$settingsJson.comfy_root)
            }
        } catch {
            Write-NexusLine "Could not read configured ComfyUI path; falling back to embedded runtime." "Warn"
        }
    }
    return Join-Path $root "runtime\ComfyUI"
}

function Get-NexusConfiguredComfyPython {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_COMFY_PYTHON) -and (Test-Path -LiteralPath $env:NEXUS_COMFY_PYTHON)) {
        return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:NEXUS_COMFY_PYTHON)
    }
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            if (![string]::IsNullOrWhiteSpace([string]$settingsJson.comfy_python) -and (Test-Path -LiteralPath ([string]$settingsJson.comfy_python))) {
                return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([string]$settingsJson.comfy_python)
            }
        } catch {
            Write-NexusLine "Could not read configured ComfyUI Python; falling back to Nexus runtime Python." "Warn"
        }
    }
    return $python
}

function Get-NexusConfiguredCustomNodesDir {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_CUSTOM_NODES_DIR)) {
        return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($env:NEXUS_CUSTOM_NODES_DIR)
    }
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            if (![string]::IsNullOrWhiteSpace([string]$settingsJson.custom_nodes_dir)) {
                return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([string]$settingsJson.custom_nodes_dir)
            }
        } catch {
            Write-NexusLine "Could not read configured custom nodes path; falling back to ./custom_nodes." "Warn"
        }
    }
    return Join-Path $root "custom_nodes"
}

function Update-NexusComfyCoreSafely {
    param([string]$ComfyRoot)

    if (!(Test-Path -LiteralPath (Join-Path $ComfyRoot ".git"))) {
        Write-NexusLine "ComfyUI is not a Git checkout; skipping core update." "Warn"
        return $false
    }
    $dirty = @(git -C $ComfyRoot status --porcelain)
    if ($dirty.Count -gt 0) {
        Write-NexusLine "ComfyUI has local changes; preserving them and skipping the core update. Commit/stash them, then rerun update_nexus.ps1." "Warn"
        return $false
    }
    git -C $ComfyRoot fetch origin --prune
    if ($LASTEXITCODE -ne 0) { throw "Could not fetch ComfyUI origin." }
    $counts = (git -C $ComfyRoot rev-list --left-right --count HEAD...origin/master).Trim().Split("`t")
    $ahead = [int]$counts[0]
    $behind = [int]$counts[1]
    if ($ahead -gt 0) {
        Write-NexusLine "ComfyUI has $ahead local commit(s); preserving them and skipping fast-forward update." "Warn"
        return $false
    }
    if ($behind -eq 0) {
        Write-NexusLine "ComfyUI core is already current." "Ok"
        return $true
    }
    git -C $ComfyRoot pull --ff-only origin master
    if ($LASTEXITCODE -ne 0) { throw "ComfyUI fast-forward update failed." }
    Write-NexusLine "ComfyUI core updated by $behind commit(s)." "Ok"
    return $true
}

$comfyRoot = Get-NexusConfiguredComfyRoot
$comfyPython = Get-NexusConfiguredComfyPython
$customNodesDir = Get-NexusConfiguredCustomNodesDir

function Invoke-NexusPipInstall {
    param(
        [string]$Label,
        [string]$PythonExe,
        [string[]]$PipArgs
    )

    $output = & $PythonExe -m pip install --disable-pip-version-check -q @PipArgs 2>&1
    if ($LASTEXITCODE -ne 0) {
        $text = ($output | Out-String).Trim()
        if (![string]::IsNullOrWhiteSpace($text)) {
            Write-NexusLine "$Label pip output:" "Error"
            Write-Host $text
        }
        throw "$Label pip install failed with exit code $LASTEXITCODE"
    }
}

Write-NexusLogo
Write-NexusSection "Updates"
Invoke-NexusRepositoryUpdate -ProjectRoot $root -Strict

if (!(Test-Path -LiteralPath (Join-Path $comfyRoot "main.py")) -or !(Test-Path -LiteralPath $python)) {
    if ((!(Test-Path -LiteralPath (Join-Path $comfyRoot "main.py"))) -and !([string]$env:NEXUS_DOWNLOAD_COMFY_RUNTIME -match '^(1|true|yes|y)$')) {
        throw "ComfyUI runtime not found at $comfyRoot. Use run.bat/update.bat to choose download or a custom ComfyUI path."
    }
    Write-NexusLine "Embedded ComfyUI is missing; preparing local runtime..." "Warn"
    & $bootstrap -ProjectRoot $root -CopyPythonEnv
    $comfyRoot = Get-NexusConfiguredComfyRoot
    $comfyPython = Get-NexusConfiguredComfyPython
}

Write-NexusLine "Checking ComfyUI core update..." "Info"
Update-NexusComfyCoreSafely -ComfyRoot $comfyRoot | Out-Null

if (Test-Path -LiteralPath $python) {
    Write-NexusSection "Requirements"
    Write-NexusLine "Backend Python..." "Info"
    Invoke-NexusPipInstall "Backend Python pip" $python @("--upgrade", "pip")
    $nexusRequirements = Join-Path $root "requirements.txt"
    if (Test-Path -LiteralPath $nexusRequirements) {
        Invoke-NexusPipInstall "Backend Python requirements" $python @("-r", $nexusRequirements)
    } else {
        Invoke-NexusPipInstall "Backend Python requirements" $python @("uvicorn[standard]>=0.30", "fastapi", "pydantic", "python-multipart", "httpx", "websockets", "pillow", "soundfile", "opencv-contrib-python")
    }
    $env:PYTHONPATH = Join-Path $root "backend"
    $backendProbe = & $python -c "import nexus_backend.main; print('backend import ok')" 2>&1
    if ($LASTEXITCODE -ne 0) {
        $text = ($backendProbe | Out-String).Trim()
        if (![string]::IsNullOrWhiteSpace($text)) {
            Write-NexusLine "Backend import probe output:" "Error"
            Write-Host $text
        }
        throw "Backend import probe failed after dependency update."
    }
    Write-NexusLine "Backend Python requirements satisfied." "Ok"

    $comfyRequirements = Join-Path $comfyRoot "requirements.txt"
    if (Test-Path -LiteralPath $comfyRequirements) {
        Write-NexusLine "ComfyUI Python..." "Info"
        Invoke-NexusPipInstall "ComfyUI Python requirements" $comfyPython @("-r", $comfyRequirements)
        Write-NexusLine "ComfyUI Python requirements satisfied." "Ok"
    }

    if (Test-Path -LiteralPath $customNodeDeps) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $customNodeDeps -ProjectRoot $root -RuntimePython $comfyPython -CustomNodesDir $customNodesDir
    }

    if (Test-Path -LiteralPath $ltxDirectorDeps) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $ltxDirectorDeps -ProjectRoot $root -RuntimePython $comfyPython -ModelsDir $modelsDir -CustomNodesDir $customNodesDir -ComfyRoot $comfyRoot -Strict
    }

    if (Test-Path -LiteralPath $wan22Deps) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $wan22Deps -ProjectRoot $root -RuntimePython $comfyPython -ModelsDir $modelsDir -Strict
    }

    if (Test-Path -LiteralPath $minimaxH3Deps) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $minimaxH3Deps -ProjectRoot $root -RuntimePython $comfyPython -ComfyRoot $comfyRoot -CustomNodesDir $customNodesDir
        if ($LASTEXITCODE -eq 2) {
            Write-NexusLine "MiniMax H3 stays disabled until the protected ComfyUI core update can be completed." "Warn"
        } elseif ($LASTEXITCODE -ne 0) {
            throw "MiniMax H3 dependency check failed."
        }
    }

    if (Test-Path -LiteralPath $minimaxH3Workflows) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $minimaxH3Workflows -ProjectRoot $root
        if ($LASTEXITCODE -ne 0) { throw "MiniMax H3 workflow import failed." }
    }

} else {
    Write-NexusLine "Runtime Python not found; run run.bat after bootstrap." "Warn"
}

Write-NexusSection "Folders"
New-Item -ItemType Directory -Force -Path (Join-Path $root "workflows\nexus_base") | Out-Null
foreach ($dir in @(
    "checkpoints\sd15",
    "checkpoints\sdxl",
    "checkpoints\flux",
    "checkpoints\qwen",
    "checkpoints\lumina",
    "checkpoints\wan",
    "checkpoints\ltx",
    "checkpoints\anima",
    "checkpoints\ideogram4",
    "diffusion_models\ideogram4",
    "diffusion_models\minimax_h3",
    "loras",
    "vae",
    "vae\minimax_h3",
    "text_encoders",
    "text_encoders\minimax_h3",
    "clip_vision",
    "controlnet",
    "upscale_models",
    "latent_upscale_models",
    "embeddings"
)) {
    New-Item -ItemType Directory -Force -Path (Join-Path $modelsDir $dir) | Out-Null
}

Write-NexusLine "Model and workflow folders are ready." "Ok"
Write-NexusLine "Verification complete." "Ok"
