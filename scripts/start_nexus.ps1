param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [switch]$StartComfy,
    [switch]$RequireComfy,
    [int]$ComfyWarmupSeconds = 75,
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$python = Join-Path $root "runtime\.venv\Scripts\python.exe"
$backend = Join-Path $root "backend\run_backend.py"
$comfyMain = Join-Path $root "runtime\ComfyUI\main.py"
$comfyRoot = Join-Path $root "runtime\ComfyUI"
$bootstrap = Join-Path $root "scripts\bootstrap_nexus_runtime.ps1"
$watcher = Join-Path $root "scripts\watch_launcher.ps1"
$runtimeHotfixes = Join-Path $root "scripts\apply_runtime_hotfixes.ps1"
$customNodeDeps = Join-Path $root "scripts\install_comfy_custom_node_deps.ps1"
$ltxDirectorDeps = Join-Path $root "scripts\install_ltx_director_deps.ps1"
$wan22Deps = Join-Path $root "scripts\install_wan22_deps.ps1"
$dinov3Deps = Join-Path $root "scripts\install_dinov3_deps.ps1"
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
$settingsPath = Join-Path $root "config\nexus_settings.json"
$dependencyStatePath = Join-Path $root "config\nexus_dependency_state.json"
$uiUrl = "http://127.0.0.1:7861/ui"

if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLogo { Write-Host "[NEXUS BTA]" }
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
    function Write-NexusSection([string]$Title) { Write-Host ""; Write-NexusLine $Title "Step" }
    function Invoke-NexusRepositoryUpdate([string]$ProjectRoot, [switch]$PromptBeforePull) { return }
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

$comfyRoot = Get-NexusConfiguredComfyRoot
$comfyMain = Join-Path $comfyRoot "main.py"
$comfyPython = Get-NexusConfiguredComfyPython
$customNodesDir = Get-NexusConfiguredCustomNodesDir

function Test-NexusHealth {
    try {
        $health = Invoke-RestMethod "http://127.0.0.1:7861/api/health" -TimeoutSec 2
        return $health.nexus -eq "ok"
    } catch {
        return $false
    }
}

function Wait-NexusHealth {
    param([int]$Seconds = 180)
    $deadline = (Get-Date).AddSeconds($Seconds)
    while ((Get-Date) -lt $deadline) {
        if (Test-NexusHealth) {
            return $true
        }
        Start-Sleep -Milliseconds 750
    }
    return $false
}

function Test-NexusComfyDirect {
    try {
        $stats = Invoke-RestMethod "http://127.0.0.1:8189/system_stats" -TimeoutSec 5
        return ($null -ne $stats)
    } catch {
        return $false
    }
}

function Test-NexusComfyHealth {
    try {
        $runtimeHealth = Invoke-RestMethod "http://127.0.0.1:7861/api/health" -TimeoutSec 5
        return [bool]$runtimeHealth.comfy_running
    } catch {
        return $false
    }
}

function Wait-NexusComfyReady {
    param([int]$Seconds = 75)

    $deadline = (Get-Date).AddSeconds($Seconds)
    $lastError = ""
    $lastNotice = Get-Date

    try {
        Invoke-RestMethod -Method Post "http://127.0.0.1:7861/api/comfy/start?wait=false" -TimeoutSec 10 | Out-Null
    } catch {
        $lastError = $_.Exception.Message
        Write-NexusLine "ComfyUI did not accept background warmup yet; continuing with readiness checks..." "Warn"
    }

    while ((Get-Date) -lt $deadline) {
        if (Test-NexusComfyHealth) {
            return $true
        }
        if (Test-NexusComfyDirect) {
            return $true
        }
        if (((Get-Date) - $lastNotice).TotalSeconds -ge 15) {
            Write-NexusLine "ComfyUI is still warming up in the background..." "Info"
            $lastNotice = Get-Date
        }
        Start-Sleep -Seconds 2
    }

    if (![string]::IsNullOrWhiteSpace($lastError)) {
        Write-NexusLine "Last ComfyUI startup response: $lastError" "Warn"
    }
    Write-NexusLine "ComfyUI is not ready yet. Nexus will open now; the first generation may wait for ComfyUI or show the detailed error." "Warn"
    Write-NexusLine "ComfyUI startup log: logs\comfyui.log" "Info"
    return $false
}

function Get-NexusFileFingerprint([string]$PathValue) {
    if (!(Test-Path -LiteralPath $PathValue)) {
        return "missing:$PathValue"
    }
    $item = Get-Item -LiteralPath $PathValue -ErrorAction SilentlyContinue
    if (!$item) {
        return "missing:$PathValue"
    }
    return "$($item.FullName)|$($item.Length)|$($item.LastWriteTimeUtc.Ticks)"
}

function Get-NexusDependencySignature {
    $parts = New-Object System.Collections.Generic.List[string]
    $parts.Add("python=$comfyPython")
    $parts.Add("models=$modelsDir")
    $parts.Add("custom_nodes=$customNodesDir")
    foreach ($pathValue in @($customNodeDeps, $ltxDirectorDeps, $wan22Deps, $dinov3Deps)) {
        $parts.Add((Get-NexusFileFingerprint $pathValue))
    }
    if (Test-Path -LiteralPath $customNodesDir) {
        $requirements = Get-ChildItem -LiteralPath $customNodesDir -Directory -ErrorAction SilentlyContinue |
            ForEach-Object { Join-Path $_.FullName "requirements.txt" } |
            Where-Object { Test-Path -LiteralPath $_ } |
            Sort-Object
        foreach ($requirementsPath in $requirements) {
            $parts.Add((Get-NexusFileFingerprint $requirementsPath))
        }
    }
    $raw = [string]::Join("`n", $parts)
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes($raw)
        return [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Test-NexusDependencyCheckRequired([string]$Signature) {
    if ([string]$env:NEXUS_FORCE_DEP_CHECK -match '^(1|true|yes|y)$') {
        return $true
    }
    if (!(Test-Path -LiteralPath $dependencyStatePath)) {
        return $true
    }
    try {
        $state = Get-Content -LiteralPath $dependencyStatePath -Raw -Encoding UTF8 | ConvertFrom-Json
        if ([string]$state.signature -ne $Signature) {
            Write-NexusLine "Dependency cache changed; run.bat will skip install scans. Open update.bat to repair or refresh dependencies if startup fails." "Warn"
            Save-NexusDependencyState $Signature
        }
        return $false
    } catch {
        Write-NexusLine "Dependency cache could not be read; run.bat will skip install scans. Open update.bat to repair or refresh dependencies if startup fails." "Warn"
        Save-NexusDependencyState $Signature
        return $false
    }
}

function Save-NexusDependencyState([string]$Signature) {
    $state = [ordered]@{
        signature = $Signature
        checked_at = (Get-Date).ToUniversalTime().ToString("o")
        comfy_python = $comfyPython
        models_dir = $modelsDir
        custom_nodes_dir = $customNodesDir
    }
    $json = $state | ConvertTo-Json -Depth 4
    $utf8NoBom = [System.Text.UTF8Encoding]::new($false)
    [System.IO.File]::WriteAllText($dependencyStatePath, $json, $utf8NoBom)
}

function Stop-NexusRuntimeProcesses {
    $patterns = @(
        "backend\\run_backend\.py",
        "backend/run_backend\.py",
        "runtime\\ComfyUI\\main\.py",
        "runtime/ComfyUI/main\.py"
    )

    foreach ($port in @(7861, 8189)) {
        $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
        foreach ($ownerPid in $owners) {
            if ($ownerPid -eq $PID) { continue }
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$ownerPid" -ErrorAction SilentlyContinue
            if (!$proc) { continue }
            $command = [string]$proc.CommandLine
            $belongsToNexus = $command -like "*$root*" -or ($patterns | Where-Object { $command -match $_ })
            if ($belongsToNexus) {
                Stop-Process -Id $ownerPid -Force -ErrorAction SilentlyContinue
            }
        }
    }

    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.ProcessId -ne $PID -and
            ([string]$_.CommandLine) -like "*$root*" -and
            ([string]$_.CommandLine) -match "run_backend\.py|ComfyUI\\main\.py|ComfyUI/main\.py"
        } |
        ForEach-Object {
            Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
        }
    Start-Sleep -Milliseconds 800
}

if (!$NoOpen) {
    $parent = Get-CimInstance Win32_Process -Filter "ProcessId=$PID" -ErrorAction SilentlyContinue
    if ($parent -and (Test-Path -LiteralPath $watcher)) {
        Start-Process -FilePath "powershell" -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", $watcher,
            "-LauncherPid", $parent.ParentProcessId,
            "-ProjectRoot", $root
        ) -WindowStyle Hidden
    }
}

Write-NexusLogo
Write-NexusSection "Updates"
Invoke-NexusRepositoryUpdate -ProjectRoot $root -PromptBeforePull

if (!(Test-Path -LiteralPath $comfyMain) -or !(Test-Path -LiteralPath $python)) {
    if (!(Test-Path -LiteralPath $bootstrap)) {
        throw "Runtime missing and bootstrap script not found."
    }
    if ((!(Test-Path -LiteralPath $comfyMain)) -and !([string]$env:NEXUS_DOWNLOAD_COMFY_RUNTIME -match '^(1|true|yes|y)$')) {
        throw "ComfyUI runtime not found at $comfyRoot. Run run.bat again and choose download, or select a custom ComfyUI path."
    }
    Write-NexusSection "First Run"
    Write-NexusLine "Preparing embedded ComfyUI runtime..." "Info"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrap -ProjectRoot $root -CopyPythonEnv
    $comfyRoot = Get-NexusConfiguredComfyRoot
    $comfyMain = Join-Path $comfyRoot "main.py"
    $comfyPython = Get-NexusConfiguredComfyPython
}

if (!(Test-Path -LiteralPath $comfyMain)) {
    throw "Embedded ComfyUI runtime not found at $comfyMain."
}

if (!(Test-Path -LiteralPath $python)) {
    throw "Runtime Python was not created at $python."
}

$dependencySignature = Get-NexusDependencySignature
$dependencyCheckRequired = Test-NexusDependencyCheckRequired $dependencySignature

if ($dependencyCheckRequired -and (Test-Path -LiteralPath $ltxDirectorDeps)) {
    Write-NexusSection "Requirements"
    if (Test-Path -LiteralPath $customNodeDeps) {
        & powershell -NoProfile -ExecutionPolicy Bypass -File $customNodeDeps -ProjectRoot $root -RuntimePython $comfyPython -CustomNodesDir $customNodesDir -Strict
    }
    & powershell -NoProfile -ExecutionPolicy Bypass -File $ltxDirectorDeps -ProjectRoot $root -RuntimePython $comfyPython -ModelsDir $modelsDir -CustomNodesDir $customNodesDir -Strict
}

if ($dependencyCheckRequired -and (Test-Path -LiteralPath $wan22Deps)) {
    Write-NexusSection "Wan 2.2"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $wan22Deps -ProjectRoot $root -RuntimePython $comfyPython -ModelsDir $modelsDir -Strict
}

if ($dependencyCheckRequired -and (Test-Path -LiteralPath $dinov3Deps)) {
    Write-NexusSection "DINOv3"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $dinov3Deps -ProjectRoot $root -RuntimePython $comfyPython -ModelsDir $modelsDir -Strict
}

if ($dependencyCheckRequired) {
    Save-NexusDependencyState $dependencySignature
} else {
    Write-NexusLine "Dependency checks are cached; skipping install scans for fast startup." "Ok"
}

if (Test-Path -LiteralPath $runtimeHotfixes) {
    Write-NexusLine "Applying local runtime patches..." "Info"
    & powershell -NoProfile -ExecutionPolicy Bypass -File $runtimeHotfixes -ProjectRoot $root
}

Write-NexusSection "Runtime"
Write-NexusLine "Closing old Nexus/ComfyUI processes..." "Info"
Stop-NexusRuntimeProcesses

foreach ($relative in @("output", "temp")) {
    $target = Join-Path $comfyRoot $relative
    if (Test-Path -LiteralPath $target) {
        $resolvedTarget = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($target)
        $resolvedComfyRoot = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($comfyRoot)
        if ($resolvedTarget.StartsWith($resolvedComfyRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
            Remove-Item -LiteralPath $resolvedTarget -Recurse -Force -ErrorAction SilentlyContinue
        }
    }
}

if (Test-Path -LiteralPath $comfyRoot) {
    Get-ChildItem -LiteralPath $comfyRoot -Filter *.log -File -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    $runtimeUser = Join-Path $comfyRoot "user"
    if (Test-Path -LiteralPath $runtimeUser) {
        Get-ChildItem -LiteralPath $runtimeUser -Filter *.log -File -Recurse -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
    }
}

$portOwner = Get-NetTCPConnection -LocalPort 7861 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
if ($portOwner) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$portOwner" -ErrorAction SilentlyContinue
    if ($owner.CommandLine -match "run_backend\.py|uvicorn") {
        Stop-Process -Id $portOwner -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

$comfyPortOwner = Get-NetTCPConnection -LocalPort 8189 -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty OwningProcess
if ($comfyPortOwner) {
    $owner = Get-CimInstance Win32_Process -Filter "ProcessId=$comfyPortOwner" -ErrorAction SilentlyContinue
    if ($owner.CommandLine -match "runtime\\ComfyUI\\main\.py|runtime/ComfyUI/main\.py") {
        Stop-Process -Id $comfyPortOwner -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}

if (!(Test-NexusHealth)) {
    Write-NexusLine "Starting backend..." "Info"
    Start-Process -FilePath $python -ArgumentList @($backend) -WorkingDirectory $root -WindowStyle Hidden
}

Write-NexusLine "Waiting for API..." "Info"
if (!(Wait-NexusHealth -Seconds 180)) {
    throw "Backend did not become ready at http://127.0.0.1:7861/api/health. Open update.bat to repair or refresh dependencies, then run run.bat again."
}

Write-NexusLine "Model folders and catalog are ready." "Info"
Invoke-RestMethod -Method Post "http://127.0.0.1:7861/api/model-tree" -TimeoutSec 15 | Out-Null

if ($StartComfy) {
    Write-NexusLine "Starting embedded ComfyUI..." "Info"
    if (!(Wait-NexusComfyReady -Seconds $ComfyWarmupSeconds)) {
        if ($RequireComfy) {
            throw "ComfyUI startup failed: runtime did not become ready after $ComfyWarmupSeconds seconds. Open update.bat to repair or refresh dependencies, then run run.bat again."
        }
    }
} else {
    Write-NexusLine "ComfyUI will start on demand." "Info"
}

if (!$NoOpen) {
    Write-NexusLine "Opening interface..." "Info"
    Start-Process $uiUrl
}

Write-NexusLine "Ready: $uiUrl" "Ok"
