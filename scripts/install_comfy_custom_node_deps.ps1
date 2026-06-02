param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
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
    $resolvedArgs = @($PipArgs)
    $tempRequirements = ""
    try {
        if ($PipArgs.Count -eq 2 -and $PipArgs[0] -eq "-r" -and (Test-Path -LiteralPath $PipArgs[1])) {
            $source = [string]$PipArgs[1]
            $lines = Get-Content -LiteralPath $source
            $sanitized = @()
            $changed = $false
            foreach ($line in $lines) {
                if ($line.Trim() -match '^onnxruntime-gpu(\s*(#.*)?)?$') {
                    $sanitized += "onnxruntime>=1.18 # Nexus fallback for optional onnxruntime-gpu"
                    $changed = $true
                } elseif ($line.Trim() -match '^Imath([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "OpenEXR>=3.2.0 # Nexus fallback for Imath module"
                    $changed = $true
                } elseif ($line.Trim() -match '^aiohttp(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "aiohttp>=3.14.0 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^av(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "av>=17.0.1 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^diffusers(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "diffusers>=0.38.0 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^fastapi(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "fastapi>=0.136.3 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^huggingface[-_]hub(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "huggingface-hub>=0.36.2 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^kornia(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "kornia>=0.8.2 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^numpy(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "numpy>=2.4.6 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^opencv-contrib-python(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "opencv-contrib-python>=4.10.0.84 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^opencv-python(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "opencv-python>=4.10.0.84 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^pillow(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "pillow>=12.2.0 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^protobuf(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "protobuf>=5.29.6 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^scikit[-_]image(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "scikit-image>=0.26.0 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^tokenizers(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "tokenizers>=0.22.2 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^torch(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "torch==2.10.0+cu130 # Nexus current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^torchaudio(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "torchaudio==2.10.0+cu130 # Nexus current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^torchvision(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "torchvision==0.25.0+cu130 # Nexus current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^transformers(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "transformers>=4.57.6,<5 # Nexus minimum for current Comfy/Nexus runtime"
                    $changed = $true
                } elseif ($line.Trim() -match '^inference-(cli|gpu)(\[.*\])?([<>=!~].*)?(\s*(#.*)?)?$') {
                    $sanitized += "# $line # Nexus skip: optional Roboflow inference package conflicts with current Comfy/Nexus runtime"
                    $changed = $true
                } else {
                    $sanitized += $line
                }
            }
            if ($changed) {
                $tempRequirements = Join-Path ([System.IO.Path]::GetTempPath()) ("nexus_node_requirements_{0}.txt" -f ([System.Guid]::NewGuid().ToString("N")))
                Set-Content -LiteralPath $tempRequirements -Value $sanitized -Encoding UTF8
                $resolvedArgs = @("-r", $tempRequirements)
                Write-NexusLine "$Label uses Nexus optional dependency fallback." "Info"
            }
        }

        $dryArgs = @("-m", "pip", "install", "--dry-run", "--no-input", "--disable-pip-version-check", "-q") + $resolvedArgs
        $dryOutput = & $RuntimePython @dryArgs 2>&1
        $dryText = ($dryOutput | Out-String).Trim()
        if ($LASTEXITCODE -eq 0 -and [string]::IsNullOrWhiteSpace($dryText)) {
            Write-NexusLine "$Label requirements already satisfied." "Ok"
            return
        }

        $installOutput = & $RuntimePython -m pip install --disable-pip-version-check -q @resolvedArgs 2>&1
        if ($LASTEXITCODE -ne 0) {
            $installText = ($installOutput | Out-String).Trim()
            if (![string]::IsNullOrWhiteSpace($installText)) {
                Write-NexusLine "$Label pip output:" "Warn"
                Write-Host $installText
            }
            throw "pip install failed with exit code $LASTEXITCODE"
        }
        Write-NexusLine "$Label requirements satisfied." "Ok"
    } finally {
        if (![string]::IsNullOrWhiteSpace($tempRequirements)) {
            Remove-Item -LiteralPath $tempRequirements -Force -ErrorAction SilentlyContinue
        }
    }
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
