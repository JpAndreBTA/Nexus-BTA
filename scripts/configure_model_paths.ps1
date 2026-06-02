param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$PromptDownloads
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$configDir = Join-Path $root "config"
$settingsPath = Join-Path $configDir "nexus_settings.json"
$startupPath = Join-Path $configDir "nexus_startup.json"
$envPath = Join-Path $configDir "nexus_startup_env.cmd"
$hfTokenPath = Join-Path $configDir "huggingface_token.txt"
$defaultModelsDir = Join-Path $root "models"
$defaultComfyRoot = Join-Path $root "runtime\ComfyUI"
$defaultComfyPython = Join-Path $root "runtime\.venv\Scripts\python.exe"
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"

if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
    function Write-NexusSection([string]$Title) { Write-Host ""; Write-NexusLine $Title "Step" }
}

function ConvertTo-HashtableDeep($Value) {
    if ($Value -is [System.Management.Automation.PSCustomObject]) {
        $hash = [ordered]@{}
        foreach ($property in $Value.PSObject.Properties) {
            $hash[$property.Name] = ConvertTo-HashtableDeep $property.Value
        }
        return $hash
    }
    if ($Value -is [System.Collections.IEnumerable] -and $Value -isnot [string]) {
        return @($Value | ForEach-Object { ConvertTo-HashtableDeep $_ })
    }
    return $Value
}

function Get-SettingsData {
    if (Test-Path -LiteralPath $settingsPath) {
        return ConvertTo-HashtableDeep ((Get-Content -LiteralPath $settingsPath -Raw) | ConvertFrom-Json)
    }
    return [ordered]@{}
}

function Save-SettingsData($Data) {
    New-Item -ItemType Directory -Force -Path $configDir | Out-Null
    $json = $Data | ConvertTo-Json -Depth 20
    [System.IO.File]::WriteAllText($settingsPath, $json, [System.Text.UTF8Encoding]::new($false))
}

function Add-ModelSourceDefaults($Data) {
    if (!$Data.Contains("model_sources") -or $null -eq $Data["model_sources"]) {
        $Data["model_sources"] = [ordered]@{}
    }
    foreach ($key in @("checkpoints", "vae", "loras", "controlnet", "upscale_models", "latent_upscale_models", "refine", "frame_interpolation", "video", "3d", "workflows")) {
        if (!$Data["model_sources"].Contains($key) -or $null -eq $Data["model_sources"][$key]) {
            $Data["model_sources"][$key] = @()
        }
    }
}

function Test-HasCheckpointLikeModel([string[]]$Roots) {
    $extensions = @(".safetensors", ".ckpt", ".pt", ".pth", ".bin", ".gguf")
    foreach ($rootPath in $Roots) {
        if ([string]::IsNullOrWhiteSpace($rootPath) -or !(Test-Path -LiteralPath $rootPath)) { continue }
        foreach ($relative in @("checkpoints", "unet", "diffusion_models")) {
            $candidate = Join-Path $rootPath $relative
            if (!(Test-Path -LiteralPath $candidate)) { continue }
            $hit = Get-ChildItem -LiteralPath $candidate -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object { $extensions -contains $_.Extension.ToLowerInvariant() -and $_.Length -gt 1MB } |
                Select-Object -First 1
            if ($hit) { return $true }
        }
    }
    return $false
}

function Test-HasDinoV3Model([string[]]$Roots) {
    foreach ($rootPath in $Roots) {
        if ([string]::IsNullOrWhiteSpace($rootPath)) { continue }
        foreach ($candidate in @(
            (Join-Path $rootPath "facebook\dinov3-vitl16-pretrain-lvd1689m\model.safetensors"),
            (Join-Path $rootPath "dinov3-vitl16-pretrain-lvd1689m\model.safetensors")
        )) {
            if (Test-Path -LiteralPath $candidate) {
                $item = Get-Item -LiteralPath $candidate -ErrorAction SilentlyContinue
                if ($item -and $item.Length -gt 100MB) { return $true }
            }
        }
    }
    return $false
}

function Test-ComfyRoot([string]$PathValue) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) { return $false }
    $candidate = $PathValue
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $candidate = Split-Path -Parent $candidate
    }
    return (Test-Path -LiteralPath (Join-Path $candidate "main.py"))
}

function Resolve-ComfyRootPath([string]$PathValue) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) { return "" }
    $resolved = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
    if (Test-Path -LiteralPath $resolved -PathType Leaf) {
        $resolved = Split-Path -Parent $resolved
    }
    return $resolved
}

function Get-ComfyPythonCandidate([string]$ComfyRoot) {
    $candidates = @(
        (Join-Path $ComfyRoot ".venv\Scripts\python.exe"),
        (Join-Path (Split-Path -Parent $ComfyRoot) "python_embeded\python.exe"),
        (Join-Path (Split-Path -Parent $ComfyRoot) ".venv\Scripts\python.exe"),
        $defaultComfyPython
    )
    foreach ($candidate in $candidates) {
        if (![string]::IsNullOrWhiteSpace($candidate) -and (Test-Path -LiteralPath $candidate)) {
            return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($candidate)
        }
    }
    return $defaultComfyPython
}

function Get-DefaultCustomNodesDir([string]$ComfyRoot) {
    if (Test-SamePath $ComfyRoot $defaultComfyRoot) {
        return Join-Path $root "custom_nodes"
    }
    return Join-Path $ComfyRoot "custom_nodes"
}

function Resolve-CustomNodesPath([string]$PathValue) {
    if ([string]::IsNullOrWhiteSpace($PathValue)) { return "" }
    $resolved = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
    if ((Split-Path -Leaf $resolved) -ieq "ComfyUI" -or (Test-Path -LiteralPath (Join-Path $resolved "main.py"))) {
        $resolved = Join-Path $resolved "custom_nodes"
    } elseif ((Split-Path -Leaf $resolved) -ine "custom_nodes") {
        $resolved = Join-Path $resolved "custom_nodes"
    }
    return $resolved
}

function Test-SamePath([string]$Left, [string]$Right) {
    if ([string]::IsNullOrWhiteSpace($Left) -or [string]::IsNullOrWhiteSpace($Right)) { return $false }
    return [System.IO.Path]::GetFullPath($Left).TrimEnd("\") -eq [System.IO.Path]::GetFullPath($Right).TrimEnd("\")
}

Write-NexusSection "Model Paths"
$settings = Get-SettingsData
Add-ModelSourceDefaults $settings

if (!$settings.Contains("project_root")) { $settings["project_root"] = $root }
if (!$settings.Contains("models_dir") -or [string]::IsNullOrWhiteSpace([string]$settings["models_dir"])) {
    $settings["models_dir"] = $defaultModelsDir
}

$configuredModelsDir = [string]$settings["models_dir"]
$hasStartupChoice = Test-Path -LiteralPath $startupPath
$startupState = [ordered]@{}
if ($hasStartupChoice) {
    try {
        $startupState = ConvertTo-HashtableDeep ((Get-Content -LiteralPath $startupPath -Raw) | ConvertFrom-Json)
    } catch {
        $startupState = [ordered]@{}
    }
}

Write-NexusSection "ComfyUI Backend"
$configuredComfyRoot = if ($settings.Contains("comfy_root") -and ![string]::IsNullOrWhiteSpace([string]$settings["comfy_root"])) {
    [string]$settings["comfy_root"]
} else {
    $defaultComfyRoot
}
$configuredComfyRoot = Resolve-ComfyRootPath $configuredComfyRoot
$comfyPathPrompted = $startupState.Contains("comfy_path_prompted")
$comfyRootValid = Test-ComfyRoot $configuredComfyRoot

if (!$comfyRootValid -or !$comfyPathPrompted) {
    if ($comfyRootValid) {
        Write-NexusLine "Configured ComfyUI found: $configuredComfyRoot" "Ok"
        $choice = Read-Host "Use this ComfyUI backend? [Y/default, C/custom, D/download embedded]"
    } elseif (Test-ComfyRoot $defaultComfyRoot) {
        Write-NexusLine "Embedded ComfyUI found: $defaultComfyRoot" "Ok"
        $choice = Read-Host "Use embedded ComfyUI backend? [Y/default, C/custom]"
    } else {
        Write-NexusLine "ComfyUI was not found. Choose a custom ComfyUI folder or allow Nexus to download the embedded runtime." "Warn"
        $choice = Read-Host "ComfyUI backend [D/download embedded, C/custom]"
    }

    if ($choice -match '^(c|custom)$') {
        $customComfy = Read-Host "Custom ComfyUI folder path (folder containing main.py)"
        $customComfy = Resolve-ComfyRootPath $customComfy
        if (!(Test-ComfyRoot $customComfy)) {
            throw "ComfyUI main.py not found in custom path: $customComfy"
        }
        $configuredComfyRoot = $customComfy
        $startupState["download_comfy_runtime"] = $false
        Write-NexusLine "Custom ComfyUI backend selected: $configuredComfyRoot" "Ok"
    } elseif ($choice -match '^(d|download)$') {
        $configuredComfyRoot = $defaultComfyRoot
        $startupState["download_comfy_runtime"] = $true
        Write-NexusLine "Embedded ComfyUI download/bootstrap approved." "Ok"
    } elseif ($comfyRootValid) {
        Write-NexusLine "Using configured ComfyUI backend: $configuredComfyRoot" "Ok"
    } elseif (Test-ComfyRoot $defaultComfyRoot) {
        $configuredComfyRoot = $defaultComfyRoot
        Write-NexusLine "Using embedded ComfyUI backend." "Ok"
    } else {
        $configuredComfyRoot = $defaultComfyRoot
        $startupState["download_comfy_runtime"] = $true
        Write-NexusLine "Embedded ComfyUI download/bootstrap approved." "Ok"
    }

    $startupState["comfy_path_prompted"] = $true
    $startupState["selected_comfy_root"] = $configuredComfyRoot
    $startupState["comfy_choice_saved_at"] = (Get-Date).ToString("s")
} else {
    Write-NexusLine "Using configured ComfyUI backend: $configuredComfyRoot" "Ok"
    $startupState["selected_comfy_root"] = $configuredComfyRoot
}

$configuredComfyPython = if ($settings.Contains("comfy_python") -and ![string]::IsNullOrWhiteSpace([string]$settings["comfy_python"]) -and (Test-Path -LiteralPath ([string]$settings["comfy_python"]))) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([string]$settings["comfy_python"])
} else {
    Get-ComfyPythonCandidate $configuredComfyRoot
}
$defaultCustomNodesDir = Get-DefaultCustomNodesDir $configuredComfyRoot
$configuredCustomNodesDir = if ($settings.Contains("custom_nodes_dir") -and ![string]::IsNullOrWhiteSpace([string]$settings["custom_nodes_dir"])) {
    Resolve-CustomNodesPath ([string]$settings["custom_nodes_dir"])
} else {
    $defaultCustomNodesDir
}
$customNodesPathPrompted = $startupState.Contains("custom_nodes_path_prompted")
if (!$customNodesPathPrompted -or [string]::IsNullOrWhiteSpace($configuredCustomNodesDir)) {
    Write-NexusLine "Choose where Nexus should install and validate ComfyUI custom nodes." "Info"
    Write-NexusLine "Default: $defaultCustomNodesDir" "Info"
    $choice = Read-Host "Use default custom_nodes path? [Y/default, C/custom]"
    if ($choice -match '^(c|custom)$') {
        $customPath = Read-Host "Custom nodes folder path (folder named custom_nodes, or a ComfyUI folder)"
        if (![string]::IsNullOrWhiteSpace($customPath)) {
            $configuredCustomNodesDir = Resolve-CustomNodesPath $customPath
            Write-NexusLine "Custom nodes path selected: $configuredCustomNodesDir" "Ok"
        }
    } else {
        $configuredCustomNodesDir = $defaultCustomNodesDir
        Write-NexusLine "Default custom nodes path selected." "Ok"
    }
    $startupState["custom_nodes_path_prompted"] = $true
    $startupState["selected_custom_nodes_dir"] = $configuredCustomNodesDir
    $startupState["custom_nodes_choice_saved_at"] = (Get-Date).ToString("s")
} else {
    Write-NexusLine "Using configured custom nodes path: $configuredCustomNodesDir" "Ok"
}
New-Item -ItemType Directory -Force -Path $configuredCustomNodesDir | Out-Null

$settings["comfy_root"] = $configuredComfyRoot
$settings["comfy_python"] = $configuredComfyPython
$settings["custom_nodes_dir"] = $configuredCustomNodesDir
$settings["runtime_dir"] = Join-Path $root "runtime"
Save-SettingsData $settings

$usingCustomFromSettings = [System.IO.Path]::GetFullPath($configuredModelsDir).TrimEnd("\") -ne [System.IO.Path]::GetFullPath($defaultModelsDir).TrimEnd("\")

if (!$hasStartupChoice -and !$usingCustomFromSettings) {
    Write-NexusLine "Choose the model folder for this installation." "Info"
    Write-NexusLine "Default: $defaultModelsDir" "Info"
    $choice = Read-Host "Use default model path? [Y/default, C/custom]"
    if ($choice -match '^(c|custom)$') {
        $customPath = Read-Host "Custom model folder path"
        if (![string]::IsNullOrWhiteSpace($customPath)) {
            $configuredModelsDir = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($customPath)
            $settings["models_dir"] = $configuredModelsDir
            Write-NexusLine "Custom model path selected: $configuredModelsDir" "Ok"
        }
    } else {
        $configuredModelsDir = $defaultModelsDir
        $settings["models_dir"] = $configuredModelsDir
        Write-NexusLine "Default model path selected." "Ok"
    }
    Save-SettingsData $settings
    $startupState["model_path_prompted"] = $true
    $startupState["selected_models_dir"] = $configuredModelsDir
    $startupState["saved_at"] = (Get-Date).ToString("s")
} else {
    Write-NexusLine "Using configured model path: $configuredModelsDir" "Ok"
}

New-Item -ItemType Directory -Force -Path $configuredModelsDir | Out-Null
foreach ($dir in @("checkpoints", "unet", "diffusion_models", "loras", "vae", "text_encoders", "clip_vision", "controlnet", "3d")) {
    New-Item -ItemType Directory -Force -Path (Join-Path $configuredModelsDir $dir) | Out-Null
}

$roots = @($configuredModelsDir)
foreach ($key in @("checkpoints", "unet", "diffusion_models")) {
    foreach ($item in @($settings["model_sources"][$key])) {
        if (![string]::IsNullOrWhiteSpace([string]$item)) { $roots += [string]$item }
    }
}
foreach ($key in @("3d", "clip_vision")) {
    foreach ($item in @($settings["model_sources"][$key])) {
        if (![string]::IsNullOrWhiteSpace([string]$item)) { $roots += [string]$item }
    }
}

$allowModelDownloads = "0"
$missingStartupModels = @()
if (!(Test-HasCheckpointLikeModel $roots)) { $missingStartupModels += "checkpoint/unet/diffusion" }
if (!(Test-HasDinoV3Model $roots)) { $missingStartupModels += "DINOv3 for TRELLIS.2" }
if ($missingStartupModels.Count -gt 0) {
    $savedDownloadChoiceApplies = $startupState.Contains("selected_models_dir") -and
        ([System.IO.Path]::GetFullPath([string]$startupState["selected_models_dir"]).TrimEnd("\") -eq [System.IO.Path]::GetFullPath($configuredModelsDir).TrimEnd("\")) -and
        $startupState.Contains("download_optional_models")
    $shouldPromptDownloads = $PromptDownloads -or !$hasStartupChoice
    if ($savedDownloadChoiceApplies -and !$shouldPromptDownloads) {
        $allowModelDownloads = if ([bool]$startupState["download_optional_models"]) { "1" } else { "0" }
        if ($allowModelDownloads -eq "1") {
            Write-NexusLine "Saved choice allows optional heavy model downloads for this model path." "Info"
        } else {
            Write-NexusLine "Saved choice skips optional heavy model downloads for this model path." "Info"
        }
    } elseif ($shouldPromptDownloads) {
        Write-NexusLine ("Missing optional heavy model assets in configured paths: {0}" -f ($missingStartupModels -join ", ")) "Warn"
        $downloadChoice = Read-Host "Download optional heavy model assets now? [y/N]"
        if ($downloadChoice -match '^(y|yes)$') {
            $allowModelDownloads = "1"
            $startupState["download_optional_models"] = $true
        } else {
            $startupState["download_optional_models"] = $false
            Write-NexusLine "Optional heavy model downloads will be skipped. You can add models later in Settings or the selected folder." "Info"
        }
        $startupState["selected_models_dir"] = $configuredModelsDir
        $startupState["download_choice_saved_at"] = (Get-Date).ToString("s")
    } else {
        $allowModelDownloads = "0"
        $startupState["download_optional_models"] = $false
        $startupState["selected_models_dir"] = $configuredModelsDir
        $startupState["download_choice_saved_at"] = (Get-Date).ToString("s")
        Write-NexusLine "Optional heavy model downloads are disabled for run.bat. Use update.bat to change this choice." "Info"
    }
} else {
    $startupState["selected_models_dir"] = $configuredModelsDir
}

$startupState["selected_comfy_root"] = $configuredComfyRoot
$startupState["selected_custom_nodes_dir"] = $configuredCustomNodesDir
$startupState["last_checked_at"] = (Get-Date).ToString("s")
$startupJson = $startupState | ConvertTo-Json -Depth 10
[System.IO.File]::WriteAllText($startupPath, $startupJson, [System.Text.UTF8Encoding]::new($false))

$hfToken = ""
if (Test-Path -LiteralPath $hfTokenPath) {
    foreach ($line in [System.IO.File]::ReadAllLines($hfTokenPath, [System.Text.Encoding]::UTF8)) {
        $value = [string]$line.Trim()
        if (![string]::IsNullOrWhiteSpace($value) -and !$value.StartsWith("#")) {
            $hfToken = $value
            break
        }
    }
}

@(
    "@echo off",
    "set ""NEXUS_MODELS_DIR=$configuredModelsDir""",
    "set ""NEXUS_COMFY_ROOT=$configuredComfyRoot""",
    "set ""NEXUS_COMFY_PYTHON=$configuredComfyPython""",
    "set ""NEXUS_CUSTOM_NODES_DIR=$configuredCustomNodesDir""",
    "set ""NEXUS_DOWNLOAD_COMFY_RUNTIME=$($startupState["download_comfy_runtime"])""",
    "set ""NEXUS_ALLOW_MODEL_DOWNLOADS=$allowModelDownloads""",
    "set ""HF_TOKEN=$hfToken""",
    "set ""HUGGINGFACE_HUB_TOKEN=$hfToken"""
) | Set-Content -LiteralPath $envPath -Encoding ASCII

Write-NexusLine "Model path and ComfyUI backend bootstrap ready." "Ok"
