param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RuntimePython = "",
    [string]$ComfyRoot = "",
    [string]$CustomNodesDir = ""
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
if ([string]::IsNullOrWhiteSpace($RuntimePython)) { $RuntimePython = Join-Path $root "runtime\.venv\Scripts\python.exe" }
if ([string]::IsNullOrWhiteSpace($ComfyRoot)) { $ComfyRoot = Join-Path $root "runtime\ComfyUI" }
if ([string]::IsNullOrWhiteSpace($CustomNodesDir)) {
    if (![string]::IsNullOrWhiteSpace($env:NEXUS_CUSTOM_NODES_DIR)) {
        $CustomNodesDir = $env:NEXUS_CUSTOM_NODES_DIR
    } else {
        $settingsPath = Join-Path $root "config\nexus_settings.json"
        if (Test-Path -LiteralPath $settingsPath) {
            try {
                $configured = [string]((Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json).custom_nodes_dir)
                if (![string]::IsNullOrWhiteSpace($configured)) { $CustomNodesDir = $configured }
            } catch {
                Write-Warning "Could not read custom_nodes_dir from $settingsPath; using the Nexus default."
            }
        }
        if ([string]::IsNullOrWhiteSpace($CustomNodesDir)) { $CustomNodesDir = Join-Path $root "custom_nodes" }
    }
}
$CustomNodesDir = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CustomNodesDir)

if (!(Test-Path -LiteralPath $RuntimePython)) { throw "Runtime Python not found: $RuntimePython" }
if (!(Test-Path -LiteralPath (Join-Path $ComfyRoot "main.py"))) { throw "ComfyUI root not found: $ComfyRoot" }
if (!(Test-Path -LiteralPath $CustomNodesDir)) { New-Item -ItemType Directory -Path $CustomNodesDir -Force | Out-Null }

$videoHelper = Get-ChildItem -LiteralPath $CustomNodesDir -Directory -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -ieq "ComfyUI-VideoHelperSuite" } |
    Where-Object { Test-Path -LiteralPath (Join-Path $_.FullName "__init__.py") } |
    Select-Object -First 1
if (!$videoHelper) {
    $videoHelperPath = Join-Path $CustomNodesDir "ComfyUI-VideoHelperSuite"
    Write-Host "[NEXUS BTA] Installing Video Helper Suite for optimized MiniMax H3 video references..."
    & git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git $videoHelperPath
    if ($LASTEXITCODE -ne 0) { throw "Video Helper Suite installation failed." }
    $videoHelper = Get-Item -LiteralPath $videoHelperPath
    $videoHelperRequirements = Join-Path $videoHelper.FullName "requirements.txt"
    if (Test-Path -LiteralPath $videoHelperRequirements) {
        & $RuntimePython -m pip install --disable-pip-version-check -q -r $videoHelperRequirements
        if ($LASTEXITCODE -ne 0) { throw "Video Helper Suite requirements installation failed." }
    }
} else {
    Write-Host "[NEXUS BTA] Video Helper Suite is available: $($videoHelper.FullName)"
}

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

Write-Host "[NEXUS BTA] MiniMax H3 local dependencies are ready (optimized video references; SageAttention preserved; xFormers unchanged)."
