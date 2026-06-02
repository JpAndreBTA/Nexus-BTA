param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

$ErrorActionPreference = "SilentlyContinue"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$patterns = @(
    "backend\\run_backend\.py",
    "backend/run_backend\.py",
    "runtime\\ComfyUI\\main\.py",
    "runtime/ComfyUI/main\.py"
)

foreach ($port in @(7861, 8189)) {
    $owners = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($ownerPid in $owners) {
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
        $cmd = [string]$_.CommandLine
        $cmd -like "*$root*" -and ($cmd -match "run_backend\.py|ComfyUI\\main\.py|ComfyUI/main\.py")
    } |
    ForEach-Object {
        Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }
