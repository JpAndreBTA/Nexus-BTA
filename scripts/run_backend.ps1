param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$SkipRequirementRepair
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$python = Join-Path $root "runtime\.venv\Scripts\python.exe"
$bootstrap = Join-Path $root "scripts\bootstrap_nexus_runtime.ps1"
$requirements = Join-Path $root "requirements.txt"
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"

if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
}

if (!(Test-Path -LiteralPath $python)) {
    if (Test-Path -LiteralPath $bootstrap) {
        Write-NexusLine "Backend Python runtime missing; preparing local runtime..." "Warn"
        & powershell -NoProfile -ExecutionPolicy Bypass -File $bootstrap -ProjectRoot $root -CopyPythonEnv
    }
}

if (!(Test-Path -LiteralPath $python)) {
    throw "Runtime Python was not found at $python. Install Python 3.11/3.12 and run update.bat."
}

if (!$SkipRequirementRepair -and (Test-Path -LiteralPath $requirements)) {
    $probe = & $python -c "import fastapi,uvicorn,pydantic,PIL,httpx,websockets" 2>&1
    if ($LASTEXITCODE -ne 0) {
        Write-NexusLine "Backend imports failed; repairing requirements..." "Warn"
        & $python -m pip install --disable-pip-version-check -r $requirements
        if ($LASTEXITCODE -ne 0) {
            throw "Backend requirement repair failed."
        }
    }
}

$env:PYTHONPATH = Join-Path $root "backend"
Push-Location $root
try {
    & $python ".\backend\run_backend.py"
}
finally {
    Pop-Location
}
