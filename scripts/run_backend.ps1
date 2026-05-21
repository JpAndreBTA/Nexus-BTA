param(
    [string]$ProjectRoot = "D:\NexusBTA"
)

$ErrorActionPreference = "Stop"
$python = Join-Path $ProjectRoot "runtime\.venv\Scripts\python.exe"
if (!(Test-Path -LiteralPath $python)) {
    $python = "python"
}

$env:PYTHONPATH = Join-Path $ProjectRoot "backend"
Push-Location $ProjectRoot
try {
    & $python ".\backend\run_backend.py"
}
finally {
    Pop-Location
}
