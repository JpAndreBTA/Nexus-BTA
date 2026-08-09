param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$ComfyRoot = "",
    [string]$RuntimePython = "",
    [string]$CustomNodesDir = "",
    [int]$CheckIntervalHours = 24,
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$settingsPath = Join-Path $root "config\nexus_settings.json"
$refreshStatePath = Join-Path $root "config\nexus_runtime_refresh_state.json"
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
$customNodeDeps = Join-Path $root "scripts\install_comfy_custom_node_deps.ps1"
$minimaxH3Deps = Join-Path $root "scripts\install_minimax_h3_deps.ps1"
if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
}

function Get-ConfiguredPath([string]$Current, [string]$EnvironmentName, [string]$SettingsName, [string]$Fallback) {
    if (![string]::IsNullOrWhiteSpace($Current)) { return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Current) }
    if (![string]::IsNullOrWhiteSpace([string][Environment]::GetEnvironmentVariable($EnvironmentName, "Process"))) {
        return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath([Environment]::GetEnvironmentVariable($EnvironmentName, "Process"))
    }
    if (Test-Path -LiteralPath $settingsPath) {
        try {
            $settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
            $value = [string]$settings.$SettingsName
            if (![string]::IsNullOrWhiteSpace($value)) { return $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($value) }
        } catch {
            Write-NexusLine "Could not read $SettingsName from Nexus settings; using fallback." "Warn"
        }
    }
    return $Fallback
}

$ComfyRoot = Get-ConfiguredPath $ComfyRoot "NEXUS_COMFY_ROOT" "comfy_root" (Join-Path $root "runtime\ComfyUI")
$RuntimePython = Get-ConfiguredPath $RuntimePython "NEXUS_COMFY_PYTHON" "comfy_python" (Join-Path $root "runtime\.venv\Scripts\python.exe")
$CustomNodesDir = Get-ConfiguredPath $CustomNodesDir "NEXUS_CUSTOM_NODES_DIR" "custom_nodes_dir" (Join-Path $root "custom_nodes")

function Get-PathFingerprint([string]$PathValue) {
    if (!(Test-Path -LiteralPath $PathValue)) { return "missing:$PathValue" }
    $item = Get-Item -LiteralPath $PathValue -ErrorAction SilentlyContinue
    if (!$item) { return "missing:$PathValue" }
    return "$($item.FullName)|$($item.LastWriteTimeUtc.Ticks)|$($item.Length)"
}

function Get-RefreshSignature {
    $parts = New-Object System.Collections.Generic.List[string]
    $parts.Add((Get-PathFingerprint $PSCommandPath))
    $parts.Add("comfy=$ComfyRoot")
    $parts.Add("python=$RuntimePython")
    $parts.Add("custom_nodes=$CustomNodesDir")
    $parts.Add((Get-PathFingerprint (Join-Path $ComfyRoot "requirements.txt")))
    if (Test-Path -LiteralPath $CustomNodesDir) {
        Get-ChildItem -LiteralPath $CustomNodesDir -Directory -ErrorAction SilentlyContinue | Sort-Object Name | ForEach-Object {
            $parts.Add($_.FullName)
            $parts.Add((Get-PathFingerprint (Join-Path $_.FullName "requirements.txt")))
        }
    }
    $sha = [System.Security.Cryptography.SHA256]::Create()
    try {
        $bytes = [System.Text.Encoding]::UTF8.GetBytes([string]::Join("`n", $parts))
        return [System.BitConverter]::ToString($sha.ComputeHash($bytes)).Replace("-", "").ToLowerInvariant()
    } finally {
        $sha.Dispose()
    }
}

function Get-TrackedGitPaths {
    $paths = @($ComfyRoot)
    if (Test-Path -LiteralPath $CustomNodesDir) {
        $paths += @(Get-ChildItem -LiteralPath $CustomNodesDir -Directory -ErrorAction SilentlyContinue | Select-Object -ExpandProperty FullName)
    }
    return @($paths | Where-Object { Test-Path -LiteralPath (Join-Path $_ ".git") })
}

function Get-RemoteGitHeads {
    $heads = @()
    foreach ($pathValue in Get-TrackedGitPaths) {
        $line = git -C $pathValue ls-remote origin HEAD 2>$null | Select-Object -First 1
        $revision = if ($line) { ([string]$line).Split("`t")[0].Trim() } else { "" }
        if (![string]::IsNullOrWhiteSpace($revision)) {
            $heads += [pscustomobject]@{ path = $pathValue; revision = $revision }
        }
    }
    return @($heads)
}

function Test-RemoteGitChanged($State) {
    $recorded = @($State.remote_heads)
    foreach ($head in Get-RemoteGitHeads) {
        $previous = $recorded | Where-Object { [string]$_.path -eq [string]$head.path } | Select-Object -First 1
        if (!$previous -or [string]$previous.revision -ne [string]$head.revision) {
            Write-NexusLine "Git manifest changed for $($head.path); scheduling the full refresh." "Info"
            return $true
        }
    }
    return $false
}

function Test-RefreshCache([string]$Signature) {
    if ($Force -or [string]$env:NEXUS_FORCE_RUNTIME_UPDATE -match '^(1|true|yes|y)$') { return $false }
    if (!(Test-Path -LiteralPath $refreshStatePath)) { return $false }
    try {
        $state = Get-Content -LiteralPath $refreshStatePath -Raw | ConvertFrom-Json
        $checkedAt = [datetime]::Parse([string]$state.checked_at).ToUniversalTime()
        $ageHours = (([datetime]::UtcNow - $checkedAt).TotalHours)
        if ([string]$state.signature -ne $Signature -or $ageHours -lt 0 -or $ageHours -ge [Math]::Max(1, $CheckIntervalHours)) { return $false }
        return !(Test-RemoteGitChanged $state)
    } catch {
        return $false
    }
}

function Save-RefreshCache([string]$Signature) {
    $state = [ordered]@{
        signature = $Signature
        checked_at = [datetime]::UtcNow.ToString("o")
        comfy_root = $ComfyRoot
        custom_nodes_dir = $CustomNodesDir
        interval_hours = $CheckIntervalHours
        remote_heads = @(Get-RemoteGitHeads)
    }
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $refreshStatePath) | Out-Null
    [System.IO.File]::WriteAllText($refreshStatePath, ($state | ConvertTo-Json -Depth 3), [System.Text.UTF8Encoding]::new($false))
}

$refreshSignature = Get-RefreshSignature
if (Test-RefreshCache $refreshSignature) {
    Write-NexusLine "ComfyUI/custom-node refresh is cached; skipping network and pip checks for this launch." "Ok"
    exit 0
}

function Update-GitRepositorySafely([string]$Label, [string]$PathValue) {
    if (!(Test-Path -LiteralPath (Join-Path $PathValue ".git"))) {
        Write-NexusLine "$Label is not a Git checkout; checking installed dependencies only." "Info"
        return $false
    }
    $dirty = @(git -C $PathValue status --porcelain)
    if ($dirty.Count -gt 0) {
        Write-NexusLine "$Label has local changes; preserving them and skipping Git update." "Warn"
        return $false
    }
    git -C $PathValue fetch origin --prune
    if ($LASTEXITCODE -ne 0) {
        Write-NexusLine "$Label fetch failed; keeping the installed version." "Warn"
        return $false
    }
    $upstream = (git -C $PathValue rev-parse --abbrev-ref --symbolic-full-name '@{u}' 2>$null).Trim()
    if ([string]::IsNullOrWhiteSpace($upstream)) {
        Write-NexusLine "$Label has no upstream branch; preserving the installed version." "Warn"
        return $false
    }
    $counts = (git -C $PathValue rev-list --left-right --count "HEAD...$upstream").Trim().Split("`t")
    $ahead = [int]$counts[0]
    $behind = [int]$counts[1]
    if ($ahead -gt 0) {
        Write-NexusLine "$Label has $ahead local commit(s); preserving them and skipping Git update." "Warn"
        return $false
    }
    if ($behind -eq 0) {
        Write-NexusLine "$Label is current." "Ok"
        return $false
    }
    git -C $PathValue pull --ff-only
    if ($LASTEXITCODE -ne 0) {
        Write-NexusLine "$Label fast-forward update failed; keeping the installed version." "Warn"
        return $false
    }
    Write-NexusLine "$Label updated by $behind commit(s)." "Ok"
    return $true
}

function Ensure-Requirements([string]$Label, [string]$RequirementsPath) {
    if (!(Test-Path -LiteralPath $RequirementsPath) -or !(Test-Path -LiteralPath $RuntimePython)) { return }
    & $RuntimePython -m pip install --disable-pip-version-check -q -r $RequirementsPath
    if ($LASTEXITCODE -eq 0) {
        Write-NexusLine "$Label requirements are satisfied." "Ok"
    } else {
        Write-NexusLine "$Label requirements could not be refreshed; keeping the existing environment." "Warn"
    }
}

Write-NexusLine "Checking configured ComfyUI runtime: $ComfyRoot" "Info"
$coreUpdated = Update-GitRepositorySafely "ComfyUI core" $ComfyRoot
Ensure-Requirements "ComfyUI" (Join-Path $ComfyRoot "requirements.txt")

if (Test-Path -LiteralPath $CustomNodesDir) {
    Get-ChildItem -LiteralPath $CustomNodesDir -Directory -ErrorAction SilentlyContinue | ForEach-Object {
        [void](Update-GitRepositorySafely "Custom node $($_.Name)" $_.FullName)
    }
} else {
    Write-NexusLine "Configured custom nodes folder was not found: $CustomNodesDir" "Warn"
}

if (Test-Path -LiteralPath $customNodeDeps) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $customNodeDeps -ProjectRoot $root -RuntimePython $RuntimePython -CustomNodesDir $CustomNodesDir
    if ($LASTEXITCODE -ne 0) { Write-NexusLine "One or more custom-node dependency checks failed; see the output above." "Warn" }
}

if (Test-Path -LiteralPath $minimaxH3Deps) {
    & powershell -NoProfile -ExecutionPolicy Bypass -File $minimaxH3Deps -ProjectRoot $root -RuntimePython $RuntimePython -ComfyRoot $ComfyRoot -CustomNodesDir $CustomNodesDir
    if ($LASTEXITCODE -eq 2) {
        Write-NexusLine "MiniMax H3 remains gated until this configured ComfyUI core contains the official H3 nodes." "Warn"
    } elseif ($LASTEXITCODE -ne 0) {
        Write-NexusLine "MiniMax H3 dependency check failed; keeping the existing runtime." "Warn"
    }
}

if ($coreUpdated) { Write-NexusLine "ComfyUI core changed; the launcher will refresh its dependency state before starting." "Info" }
Save-RefreshCache $refreshSignature
