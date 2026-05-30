param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$RuntimePython = "",
    [string]$CustomNodesDir = "",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

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

if ([string]::IsNullOrWhiteSpace($CustomNodesDir)) {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_CUSTOM_NODES_DIR)) {
        $CustomNodesDir = $env:NEXUS_CUSTOM_NODES_DIR
    } else {
        $CustomNodesDir = Join-Path $root "custom_nodes"
    }
}
$CustomNodesDir = Get-AbsolutePath $CustomNodesDir

if ([string]::IsNullOrWhiteSpace($RuntimePython)) {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_COMFY_PYTHON) -and (Test-Path -LiteralPath $env:NEXUS_COMFY_PYTHON)) {
        $RuntimePython = $env:NEXUS_COMFY_PYTHON
    } else {
        $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
        $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
    }
}

if (!(Test-Path -LiteralPath $CustomNodesDir)) {
    Write-NexusLine "Custom nodes folder not found; dependency scan skipped: $CustomNodesDir" "Warn"
    return
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

$requirements = Get-ChildItem -LiteralPath $CustomNodesDir -Directory -ErrorAction SilentlyContinue |
    ForEach-Object { Join-Path $_.FullName "requirements.txt" } |
    Where-Object { Test-Path -LiteralPath $_ }

if (!$requirements -or @($requirements).Count -eq 0) {
    Write-NexusLine "No custom node requirements found in $CustomNodesDir." "Info"
    return
}

foreach ($requirementsPath in $requirements) {
    $nodeName = Split-Path (Split-Path -Parent $requirementsPath) -Leaf
    try {
        Write-NexusLine "$nodeName Python requirements..." "Info"
        Invoke-NexusPipInstallIfNeeded $nodeName @("-r", $requirementsPath)
    } catch {
        if ($Strict) { throw }
        Write-NexusLine "$nodeName requirements failed: $($_.Exception.Message)" "Warn"
    }
}

Write-NexusLine "Custom node dependency scan complete." "Ok"
