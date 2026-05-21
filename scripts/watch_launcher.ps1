param(
    [Parameter(Mandatory = $true)]
    [int]$LauncherPid,
    [string]$ProjectRoot = "D:\NexusBTA"
)

$ErrorActionPreference = "SilentlyContinue"

while ($true) {
    $launcher = Get-Process -Id $LauncherPid -ErrorAction SilentlyContinue
    if (!$launcher) {
        $stopScript = Join-Path $ProjectRoot "scripts\stop_nexus.ps1"
        powershell -NoProfile -ExecutionPolicy Bypass -File $stopScript -ProjectRoot $ProjectRoot
        break
    }
    Start-Sleep -Seconds 2
}

