param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [switch]$NoOpen
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$python = Join-Path $root "runtime\.venv\Scripts\python.exe"
$backend = Join-Path $root "backend\run_backend.py"
$comfyMain = Join-Path $root "runtime\ComfyUI\main.py"
$comfyRoot = Join-Path $root "runtime\ComfyUI"
$watcher = Join-Path $root "scripts\watch_launcher.ps1"
$runtimeHotfixes = Join-Path $root "scripts\apply_runtime_hotfixes.ps1"
$uiUrl = "http://127.0.0.1:7861/ui"

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

if (!(Test-Path -LiteralPath $comfyMain)) {
    throw "Embedded ComfyUI runtime not found at $comfyMain. Run scripts\bootstrap_nexus_runtime.ps1 first."
}

if (!(Test-Path -LiteralPath $python)) {
    $python = "python"
}

if (Test-Path -LiteralPath $runtimeHotfixes) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $runtimeHotfixes -ProjectRoot $root
}

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
    Write-Host "[NEXUS BTA] Starting backend..."
    Start-Process -FilePath $python -ArgumentList @($backend) -WorkingDirectory $root -NoNewWindow
}

Write-Host "[NEXUS BTA] Waiting for API..."
if (!(Wait-NexusHealth -Seconds 180)) {
    throw "Backend did not become ready at http://127.0.0.1:7861/api/health"
}

Write-Host "[NEXUS BTA] Preparing model folders..."
Invoke-RestMethod -Method Post "http://127.0.0.1:7861/api/model-tree" -TimeoutSec 15 | Out-Null

Write-Host "[NEXUS BTA] Starting embedded ComfyUI runtime..."
try {
    Invoke-RestMethod -Method Post "http://127.0.0.1:7861/api/comfy/start" -TimeoutSec 180 | Out-Null
} catch {
    Write-Warning $_.Exception.Message
}

if (!$NoOpen) {
    Write-Host "[NEXUS BTA] Opening UI..."
    Start-Process $uiUrl
}

Write-Host "[NEXUS BTA] Ready: $uiUrl"
