param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RuntimePython = "",
    [string]$ComfyRoot = ""
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
if ([string]::IsNullOrWhiteSpace($RuntimePython)) { $RuntimePython = Join-Path $root "runtime\.venv\Scripts\python.exe" }
if ([string]::IsNullOrWhiteSpace($ComfyRoot)) { $ComfyRoot = Join-Path $root "runtime\ComfyUI" }

if (!(Test-Path -LiteralPath $RuntimePython)) { throw "Runtime Python not found: $RuntimePython" }
if (!(Test-Path -LiteralPath (Join-Path $ComfyRoot "main.py"))) { throw "ComfyUI root not found: $ComfyRoot" }

# Do not replace xFormers: SageAttention is an additional attention backend and
# the runtime selects it only for presets that support it.
$sageProbe = & $RuntimePython -c "import sageattention; print(getattr(sageattention, '__version__', 'installed'))" 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Host "[NEXUS BTA] Installing SageAttention for local MiniMax H3..."
    & $RuntimePython -m pip install --disable-pip-version-check sageattention
    if ($LASTEXITCODE -ne 0) { throw "SageAttention installation failed." }
} else {
    Write-Host "[NEXUS BTA] SageAttention $($sageProbe | Select-Object -Last 1) is available."
}

$coreProbe = & $RuntimePython -c "from pathlib import Path; p=Path(r'$ComfyRoot')/'comfy_extras'/'nodes_minimax_h3.py'; t=p.read_text(encoding='utf-8', errors='ignore') if p.exists() else ''; print('MiniMaxH3ImageToVideo' in t and 'MiniMaxH3ReferenceToVideo' in t)" 2>&1
if ($LASTEXITCODE -ne 0 -or ($coreProbe | Out-String).Trim() -notmatch 'True') {
    Write-Warning "This ComfyUI checkout does not yet include the MiniMax H3 core nodes. Update a clean ComfyUI checkout, then rerun this script. Existing local ComfyUI changes were not overwritten."
    exit 2
}

Write-Host "[NEXUS BTA] MiniMax H3 local dependencies are ready (SageAttention preserved; xFormers unchanged)."
