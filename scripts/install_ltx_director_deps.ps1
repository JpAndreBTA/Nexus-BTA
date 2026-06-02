param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$RuntimePython = "",
    [string]$ModelsDir = "",
    [string]$CustomNodesDir = "",
    [string]$ComfyRoot = "",
    [switch]$Strict
)

$ErrorActionPreference = "Stop"

function Write-NexusWarn([string]$Message) {
    if (Get-Command Write-NexusLine -ErrorAction SilentlyContinue) {
        Write-NexusLine $Message "Warn"
    } else {
        Write-Warning "[NEXUS BTA] $Message"
    }
}

function Invoke-NexusStep([scriptblock]$Step, [string]$Label) {
    try {
        & $Step
    } catch {
        if ($Strict) {
            throw
        }
        Write-NexusWarn "$Label failed: $($_.Exception.Message)"
    }
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

function Get-AbsolutePath([string]$PathValue) {
    $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($PathValue)
}

function Invoke-NexusPythonProbe {
    param(
        [string]$PythonExe,
        [string[]]$Arguments
    )
    $stdoutPath = Join-Path ([System.IO.Path]::GetTempPath()) ("nexus_probe_stdout_{0}.txt" -f ([System.Guid]::NewGuid().ToString("N")))
    $stderrPath = Join-Path ([System.IO.Path]::GetTempPath()) ("nexus_probe_stderr_{0}.txt" -f ([System.Guid]::NewGuid().ToString("N")))
    try {
        $quotedArguments = @($Arguments | ForEach-Object { '"' + ([string]$_).Replace('"', '\"') + '"' })
        $process = Start-Process -FilePath $PythonExe -ArgumentList $quotedArguments -NoNewWindow -Wait -PassThru -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath
        $stdout = if (Test-Path -LiteralPath $stdoutPath) { Get-Content -LiteralPath $stdoutPath -Raw -ErrorAction SilentlyContinue } else { "" }
        $stderr = if (Test-Path -LiteralPath $stderrPath) { Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue } else { "" }
        return [pscustomobject]@{
            ExitCode = $process.ExitCode
            Output = (($stdout, $stderr) -join "`n").Trim()
        }
    } finally {
        Remove-Item -LiteralPath $stdoutPath, $stderrPath -Force -ErrorAction SilentlyContinue
    }
}

$root = Get-AbsolutePath $ProjectRoot
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} elseif (!(Get-Command Write-NexusLine -ErrorAction SilentlyContinue)) {
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
}
$customNodesDir = if (![string]::IsNullOrWhiteSpace($CustomNodesDir)) {
    Get-AbsolutePath $CustomNodesDir
} elseif (![string]::IsNullOrWhiteSpace($env:NEXUS_CUSTOM_NODES_DIR)) {
    Get-AbsolutePath $env:NEXUS_CUSTOM_NODES_DIR
} else {
    Join-Path $root "custom_nodes"
}
$settingsPath = Join-Path $root "config\nexus_settings.json"
$comfyRoot = if (![string]::IsNullOrWhiteSpace($ComfyRoot)) {
    Get-AbsolutePath $ComfyRoot
} elseif (![string]::IsNullOrWhiteSpace($env:NEXUS_COMFY_ROOT)) {
    Get-AbsolutePath $env:NEXUS_COMFY_ROOT
} elseif (Test-Path -LiteralPath $settingsPath) {
    try {
        $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
        if (![string]::IsNullOrWhiteSpace([string]$settingsJson.comfy_root)) {
            Get-AbsolutePath ([string]$settingsJson.comfy_root)
        } else {
            Join-Path $root "runtime\ComfyUI"
        }
    } catch {
        Join-Path $root "runtime\ComfyUI"
    }
} else {
    Join-Path $root "runtime\ComfyUI"
}
$modelsDir = if (![string]::IsNullOrWhiteSpace($ModelsDir)) {
    Get-AbsolutePath $ModelsDir
} elseif (![string]::IsNullOrWhiteSpace($env:NEXUS_MODELS_DIR)) {
    Get-AbsolutePath $env:NEXUS_MODELS_DIR
} else {
    Join-Path $root "models"
}
$diffusionModelsDir = Join-Path $modelsDir "diffusion_models"
$melDir = Join-Path $diffusionModelsDir "MelRoFormer"
$melModel = Join-Path $melDir "MelBandRoformer_fp16.safetensors"
$ltxTransitionDir = Join-Path $modelsDir "loras\ltx_transition"
$ltxTransitionLora = Join-Path $ltxTransitionDir "ltx2.3-transition.safetensors"
$ltxIcDir = Join-Path $modelsDir "loras\ltx_ic"
$ltxUnionIcLora = Join-Path $ltxIcDir "ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors"
$ltxDetailerLora = Join-Path $ltxIcDir "ltx-2-19b-ic-lora-detailer.safetensors"
$ltxCameramanLora = Join-Path $ltxIcDir "LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"
$allowModelDownloads = [string]::IsNullOrWhiteSpace($env:NEXUS_ALLOW_MODEL_DOWNLOADS) -eq $false -and $env:NEXUS_ALLOW_MODEL_DOWNLOADS -match '^(1|true|yes|y)$'

function Get-NexusModelRootsForCategory([string]$Category) {
    $roots = @($modelsDir)
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settingsJson = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            foreach ($item in @($settingsJson.model_sources.$Category)) {
                if (![string]::IsNullOrWhiteSpace([string]$item)) { $roots += [string]$item }
            }
            foreach ($item in @($settingsJson.reference_model_sources)) {
                if (![string]::IsNullOrWhiteSpace([string]$item)) { $roots += [string]$item }
            }
        } catch {
            Write-NexusWarn "Could not read model source settings: $($_.Exception.Message)"
        }
    }
    return @($roots | Where-Object { ![string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
}

function Find-NexusLoraFile([string[]]$Names, [long]$MinBytes) {
    foreach ($rootPath in Get-NexusModelRootsForCategory "loras") {
        foreach ($folder in @("loras\ltx_ic", "loras\ltx_transition", "loras\ltx", "loras", "ltx_ic", "ltx_transition", "ltx")) {
            foreach ($name in $Names) {
                $candidate = Join-Path (Join-Path $rootPath $folder) $name
                if (Test-NexusModelFile $candidate $MinBytes) { return $candidate }
            }
        }
    }
    return ""
}

function Test-NexusModelFile([string]$PathValue, [long]$MinBytes) {
    if (!(Test-Path -LiteralPath $PathValue)) { return $false }
    return (Get-Item -LiteralPath $PathValue).Length -ge $MinBytes
}

function Test-NexusLtxVideoDecodeNode([string]$NodePath) {
    $decodeFile = Join-Path $NodePath "tiled_vae_decode.py"
    $initFile = Join-Path $NodePath "__init__.py"
    if (!(Test-Path -LiteralPath $decodeFile) -or !(Test-Path -LiteralPath $initFile)) { return $false }
    $decodeText = Get-Content -LiteralPath $decodeFile -Raw -ErrorAction SilentlyContinue
    $initText = Get-Content -LiteralPath $initFile -Raw -ErrorAction SilentlyContinue
    return $decodeText -match "class\s+LTXVTiledVAEDecode\b" -and $initText -match '"LTXVTiledVAEDecode"'
}

function Test-NexusLtxVideoRuntimeImport([string]$NodePath) {
    if ([string]::IsNullOrWhiteSpace($RuntimePython) -or !(Test-Path -LiteralPath $RuntimePython)) {
        Write-NexusWarn "Cannot validate LTXVideo runtime import because Comfy Python was not found at $RuntimePython."
        return $false
    }
    if (!(Test-Path -LiteralPath (Join-Path $comfyRoot "main.py"))) {
        Write-NexusWarn "Cannot validate LTXVideo runtime import because ComfyUI was not found at $comfyRoot."
        return $false
    }
    $probe = @'
import importlib
import importlib.util
import logging
import pathlib
import sys
import traceback

comfy_root = pathlib.Path(sys.argv[1])
custom_nodes_dir = pathlib.Path(sys.argv[2])
node_path = pathlib.Path(sys.argv[3])

sys.path.insert(0, str(comfy_root))
sys.path.insert(0, str(custom_nodes_dir))

def require_attention_runtime(label, module_name):
    try:
        if module_name == "xformers":
            import xformers.ops  # noqa: F401
            if importlib.util.find_spec("xformers._C") is None:
                raise RuntimeError("xformers._C extension is not available")
        elif module_name == "sageattention":
            importlib.import_module("triton")
            importlib.import_module("sageattention")
        else:
            importlib.import_module(module_name)
    except Exception as exc:
        print(f"{label} runtime validation failed: {type(exc).__name__}: {exc}")
        sys.exit(3)

require_attention_runtime("xFormers", "xformers")
require_attention_runtime("SageAttention", "sageattention")
# xFormers was validated above; avoid repeating its optional Triton warning when
# importing custom nodes so real node import errors stay readable.
logging.getLogger("xformers").setLevel(logging.ERROR)

try:
    module = importlib.import_module(node_path.name)
    mappings = getattr(module, "NODE_CLASS_MAPPINGS", {})
    if "LTXVTiledVAEDecode" not in mappings:
        print("ComfyUI-LTXVideo imported, but NODE_CLASS_MAPPINGS does not expose LTXVTiledVAEDecode.")
        sys.exit(2)
    sys.exit(0)
except Exception:
    traceback.print_exc()
    sys.exit(1)
'@
    $probePath = Join-Path ([System.IO.Path]::GetTempPath()) ("nexus_ltx_import_probe_{0}.py" -f ([System.Guid]::NewGuid().ToString("N")))
    try {
        [System.IO.File]::WriteAllText($probePath, $probe, [System.Text.UTF8Encoding]::new($false))
        $result = Invoke-NexusPythonProbe $RuntimePython @($probePath, $comfyRoot, $customNodesDir, $NodePath)
        if ($result.ExitCode -eq 0) {
            return $true
        }
        $text = [string]$result.Output
        if (![string]::IsNullOrWhiteSpace($text)) {
            Write-NexusWarn "ComfyUI-LTXVideo runtime import failed: $text"
        }
        return $false
    } finally {
        Remove-Item -LiteralPath $probePath -Force -ErrorAction SilentlyContinue
    }
}

function Repair-NexusRepoFolder([hashtable]$Repo) {
    if ($Repo.Name -ne "ComfyUI-LTXVideo") { return $false }
    if (Test-NexusLtxVideoDecodeNode $Repo.Path) { return $false }
    if (Test-Path -LiteralPath $Repo.Path) {
        $backup = "$($Repo.Path).stale_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
        Write-NexusWarn "$($Repo.Name) is present but missing LTXVTiledVAEDecode. Moving stale copy to $backup."
        Move-Item -LiteralPath $Repo.Path -Destination $backup -Force
    }
    Write-NexusLine "Reinstalling $($Repo.Name) with required Decode Frames node..." "Info"
    git clone --depth 1 $Repo.Url $Repo.Path
    if ($LASTEXITCODE -ne 0) {
        throw "git clone failed for $($Repo.Name)"
    }
    return $true
}

function Update-NexusGitRepo([hashtable]$Repo) {
    if (!$Repo.AutoUpdate) { return }
    $gitDir = Join-Path $Repo.Path ".git"
    if (!(Test-Path -LiteralPath $gitDir)) { return }
    if ($Repo.Commit) { return }
    Push-Location $Repo.Path
    try {
        $dirty = [string]::IsNullOrWhiteSpace((git status --porcelain)) -eq $false
        if ($dirty) {
            Write-NexusWarn "$($Repo.Name) has local changes; update skipped to avoid overwriting user edits."
            return
        }
        Write-NexusLine "Updating $($Repo.Name)..." "Info"
        git fetch --depth 1 origin
        if ($LASTEXITCODE -ne 0) {
            Write-NexusWarn "$($Repo.Name) fetch failed; keeping existing checkout."
            return
        }
        $branch = (git rev-parse --abbrev-ref HEAD).Trim()
        if ($branch -eq "HEAD" -or [string]::IsNullOrWhiteSpace($branch)) {
            git checkout --detach FETCH_HEAD
        } else {
            git merge --ff-only "origin/$branch"
            if ($LASTEXITCODE -ne 0) {
                git checkout --detach FETCH_HEAD
            }
        }
    } finally {
        Pop-Location
    }
}

function Skip-NexusModelDownload([string]$Label, [string]$PathValue) {
    if ($allowModelDownloads) { return $false }
    Write-NexusWarn "$Label is missing at $PathValue. Skipping download because startup model downloads were not approved."
    return $true
}

if (!$RuntimePython) {
    $candidate = Join-Path $root "runtime\.venv\Scripts\python.exe"
    $RuntimePython = if (Test-Path -LiteralPath $candidate) { $candidate } else { "python" }
}

New-Item -ItemType Directory -Force -Path $customNodesDir, $diffusionModelsDir, $melDir, $ltxTransitionDir, $ltxIcDir | Out-Null

function Test-NexusIsWindows {
    return [System.Runtime.InteropServices.RuntimeInformation]::IsOSPlatform([System.Runtime.InteropServices.OSPlatform]::Windows)
}

Invoke-NexusStep -Label "Installing Nexus attention runtime" -Step {
    $attentionRuntimeArgs = @(
        "--extra-index-url",
        "https://download.pytorch.org/whl/cu130",
        "sageattention>=1.0.6"
    )
    if (Test-NexusIsWindows) {
        $attentionRuntimeArgs += @(
            "xformers @ https://download.pytorch.org/whl/cu130/xformers-0.0.35-py39-none-win_amd64.whl",
            "triton-windows>=3.7.0.post26"
        )
    } else {
        $attentionRuntimeArgs += @(
            "xformers==0.0.35",
            "triton>=3.0.0"
        )
    }
    Invoke-NexusPipInstallIfNeeded "Nexus attention runtime" $attentionRuntimeArgs
}

$repos = @(
    @{
        Name = "ComfyUI-LTXVideo"
        Url = "https://github.com/Lightricks/ComfyUI-LTXVideo.git"
        Path = Join-Path $customNodesDir "ComfyUI-LTXVideo"
        AutoUpdate = $true
    },
    @{
        Name = "ComfyUI-VideoHelperSuite"
        Url = "https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git"
        Path = Join-Path $customNodesDir "ComfyUI-VideoHelperSuite"
    },
    @{
        Name = "ComfyUI-KJNodes"
        Url = "https://github.com/kijai/ComfyUI-KJNodes.git"
        Path = Join-Path $customNodesDir "comfyui-kjnodes"
    },
    @{
        Name = "rgthree-comfy"
        Url = "https://github.com/rgthree/rgthree-comfy.git"
        Path = Join-Path $customNodesDir "rgthree-comfy"
    },
    @{
        Name = "comfyui-int-and-float"
        Url = "https://github.com/danTheMonk/comfyui-int-and-float.git"
        Path = Join-Path $customNodesDir "comfyui-int-and-float"
    },
    @{
        Name = "RES4LYF"
        Url = "https://github.com/ClownsharkBatwing/RES4LYF.git"
        Path = Join-Path $customNodesDir "RES4LYF"
    },
    @{
        Name = "WhatDreamsCost-ComfyUI"
        Url = "https://github.com/WhatDreamsCost/WhatDreamsCost-ComfyUI.git"
        Path = Join-Path $customNodesDir "WhatDreamsCost-ComfyUI"
        Commit = "f0c8a322eaa607eb499ebd320f0d7d03f4caff80"
    },
    @{
        Name = "ComfyUI-MelBandRoFormer"
        Url = "https://github.com/kijai/ComfyUI-MelBandRoFormer.git"
        Path = Join-Path $customNodesDir "ComfyUI-MelBandRoFormer"
    },
    @{
        Name = "comfyui_controlnet_aux"
        Url = "https://github.com/Fannovel16/comfyui_controlnet_aux.git"
        Path = Join-Path $customNodesDir "comfyui_controlnet_aux"
    },
    @{
        Name = "ComfyUI-Video-Depth-Anything"
        Url = "https://github.com/yuvraj108c/ComfyUI-Video-Depth-Anything.git"
        Path = Join-Path $customNodesDir "ComfyUI-Video-Depth-Anything"
    }
)

foreach ($repo in $repos) {
    Invoke-NexusStep -Label "Installing $($repo.Name)" -Step {
        if ($repo.SkipClone -and !(Test-Path -LiteralPath $repo.Path)) {
            Write-NexusWarn "$($repo.Name) is not present; install it through ComfyUI Manager or place it under $customNodesDir if the workflow requires it."
            return
        }
        if (!(Test-Path -LiteralPath $repo.Path)) {
            Write-NexusLine "Installing $($repo.Name)..." "Info"
            git clone --depth 1 $repo.Url $repo.Path
            if ($LASTEXITCODE -ne 0) {
                throw "git clone failed for $($repo.Name)"
            }
        } else {
            Write-NexusLine "$($repo.Name) is present." "Ok"
        }

        Repair-NexusRepoFolder $repo | Out-Null

        if ($repo.Commit -and (Test-Path -LiteralPath (Join-Path $repo.Path ".git"))) {
            Push-Location $repo.Path
            try {
                $localDirectorDirty = $repo.Name -eq "WhatDreamsCost-ComfyUI" -and [string]::IsNullOrWhiteSpace((git status --porcelain -- "ltx_director.py")) -eq $false
                if ($localDirectorDirty) {
                    Write-NexusLine "$($repo.Name) local LTX Director hotfixes preserved; upstream checkout skipped." "Ok"
                    return
                }
                $currentCommit = (git rev-parse HEAD).Trim()
                if ($currentCommit -ne $repo.Commit) {
                    Write-NexusLine "Pinning $($repo.Name) to $($repo.Commit.Substring(0, 7))..." "Info"
                    git fetch origin $repo.Commit --depth 1
                    git checkout --detach $repo.Commit
                }
            } finally {
                Pop-Location
            }
        } else {
            Update-NexusGitRepo $repo
        }

        $requirements = Join-Path $repo.Path "requirements.txt"
        if (Test-Path -LiteralPath $requirements) {
            Write-NexusLine "$($repo.Name) Python requirements..." "Info"
            Invoke-NexusPipInstallIfNeeded $repo.Name @("-r", $requirements)
        }
    }
}

Invoke-NexusStep -Label "Validating LTXVideo Decode Frames node" -Step {
    $ltxVideoPath = Join-Path $customNodesDir "ComfyUI-LTXVideo"
    if (!(Test-NexusLtxVideoDecodeNode $ltxVideoPath)) {
        throw "ComfyUI-LTXVideo is missing LTXVTiledVAEDecode. This causes ComfyUI to report Node 'Decode Frames' not found. Run update.bat or rerun this installer with internet access."
    }
    if (!(Test-NexusLtxVideoRuntimeImport $ltxVideoPath)) {
        throw "ComfyUI-LTXVideo is installed at $ltxVideoPath, but ComfyUI Python could not import it from the configured runtime. Check the import error above, install/repair dependencies, then restart Nexus."
    }
    Write-NexusLine "ComfyUI-LTXVideo Decode Frames node is available and importable by ComfyUI." "Ok"
}

Invoke-NexusStep -Label "Downloading MelBandRoFormer model" -Step {
    if (Test-NexusModelFile $melModel 100MB) {
        Write-Host "[NEXUS BTA] MelBandRoFormer model already present."
        return
    }
    if (Skip-NexusModelDownload "MelBandRoFormer model" $melModel) {
        return
    }
    if (Test-Path -LiteralPath $melModel) {
        Remove-Item -LiteralPath $melModel -Force -ErrorAction SilentlyContinue
    }

    $url = "https://huggingface.co/Kijai/MelBandRoFormer_comfy/resolve/main/MelBandRoformer_fp16.safetensors?download=true"
    $partial = "$melModel.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Write-NexusLine "Downloading MelBandRoformer_fp16.safetensors..." "Info"
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $melModel -Force
}

Invoke-NexusStep -Label "Downloading LTX 2.3 Transition LoRA" -Step {
    $existing = Find-NexusLoraFile @("ltx2.3-transition.safetensors") 10MB
    if ($existing) {
        Write-NexusLine "LTX 2.3 Transition LoRA already present: $existing" "Ok"
        return
    }
    if (Skip-NexusModelDownload "LTX 2.3 Transition LoRA" $ltxTransitionLora) {
        return
    }
    if (Test-Path -LiteralPath $ltxTransitionLora) {
        Remove-Item -LiteralPath $ltxTransitionLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx2.3-transition.safetensors..." "Info"
    $url = "https://huggingface.co/joyfox/LTX-2.3-Transition-LORA/resolve/main/ltx2.3-transition.safetensors?download=true"
    $partial = "$ltxTransitionLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxTransitionLora -Force
}

Invoke-NexusStep -Label "Downloading LTX 2.3 IC-LoRA Union Control" -Step {
    $existing = Find-NexusLoraFile @("ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors") 100MB
    if ($existing) {
        Write-NexusLine "LTX 2.3 IC-LoRA Union Control already present: $existing" "Ok"
        return
    }
    if (Test-Path -LiteralPath $ltxUnionIcLora) {
        Remove-Item -LiteralPath $ltxUnionIcLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors..." "Info"
    $url = "https://huggingface.co/Lightricks/LTX-2.3-22b-IC-LoRA-Union-Control/resolve/main/ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors?download=true"
    $partial = "$ltxUnionIcLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxUnionIcLora -Force
}

Invoke-NexusStep -Label "Downloading LTX IC-LoRA Detailer" -Step {
    $existing = Find-NexusLoraFile @("ltx-2-19b-ic-lora-detailer.safetensors") 100MB
    if ($existing) {
        Write-NexusLine "LTX IC-LoRA Detailer already present: $existing" "Ok"
        return
    }
    if (Test-Path -LiteralPath $ltxDetailerLora) {
        Remove-Item -LiteralPath $ltxDetailerLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading ltx-2-19b-ic-lora-detailer.safetensors..." "Info"
    $url = "https://huggingface.co/Lightricks/LTX-2-19b-IC-LoRA-Detailer/resolve/main/ltx-2-19b-ic-lora-detailer.safetensors"
    $partial = "$ltxDetailerLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxDetailerLora -Force
}

Invoke-NexusStep -Label "Downloading LTX 2.3 IC-LoRA Cameraman" -Step {
    $existing = Find-NexusLoraFile @("LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors") 100MB
    if ($existing) {
        Write-NexusLine "LTX 2.3 IC-LoRA Cameraman already present: $existing" "Ok"
        return
    }
    if (Test-Path -LiteralPath $ltxCameramanLora) {
        Remove-Item -LiteralPath $ltxCameramanLora -Force -ErrorAction SilentlyContinue
    }

    Write-NexusLine "Downloading LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors..." "Info"
    $url = "https://huggingface.co/Cseti/LTX2.3-22B_IC-LoRA-Cameraman_v1/resolve/main/LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors"
    $partial = "$ltxCameramanLora.part"
    Remove-Item -LiteralPath $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -Uri $url -OutFile $partial -TimeoutSec 1800
    Move-Item -LiteralPath $partial -Destination $ltxCameramanLora -Force
}

Write-NexusLine "LTX Director requirements satisfied." "Ok"
