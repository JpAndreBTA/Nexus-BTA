param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [ValidateSet("lan", "tunnel")]
    [string]$Mode = "lan",
    [ValidateSet("ask", "tailscale", "cloudflare", "ngrok", "custom")]
    [string]$TunnelProvider = "ask",
    [int]$Port = 7861,
    [switch]$SkipComfy
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$terminalHelpers = Join-Path $root "scripts\nexus_terminal.ps1"
if (Test-Path -LiteralPath $terminalHelpers) {
    . $terminalHelpers
} else {
    function Write-NexusLogo { Write-Host "[NEXUS BTA]" }
    function Write-NexusLine([string]$Message, [string]$Kind = "Info") { Write-Host "[NEXUS BTA] $Message" }
    function Write-NexusSection([string]$Title) { Write-Host ""; Write-NexusLine $Title "Step" }
}

function Import-NexusStartupEnv {
    $startupEnv = Join-Path $root "config\nexus_startup_env.cmd"
    if (!(Test-Path -LiteralPath $startupEnv)) {
        return
    }
    cmd /c "call `"$startupEnv`" && set" | ForEach-Object {
        if ($_ -match "^(NEXUS_[^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
}

function Get-NexusLanUrls {
    $addresses = Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue |
        Where-Object {
            $_.IPAddress -and
            $_.IPAddress -ne "127.0.0.1" -and
            $_.IPAddress -notlike "169.254.*" -and
            $_.PrefixOrigin -ne "WellKnown"
        } |
        Select-Object -ExpandProperty IPAddress -Unique
    foreach ($address in $addresses) {
        "http://$address`:$Port/ui"
    }
}

function Select-NexusTunnelProvider {
    if ($TunnelProvider -ne "ask") {
        return $TunnelProvider
    }
    Write-NexusSection "Tunnel provider"
    Write-NexusLine "1. Tailscale Funnel - recommended public tunnel; requires Tailscale login/Funnel enabled." "Info"
    Write-NexusLine "2. Cloudflare Quick Tunnel - third-party relay; requires cloudflared installed." "Info"
    Write-NexusLine "3. ngrok - third-party relay; requires ngrok installed/configured." "Info"
    Write-NexusLine "4. Custom command - uses NEXUS_ONLINE_TUNNEL_COMMAND, with {port} placeholder." "Info"
    $choice = Read-Host "Choose tunnel provider [1-4]"
    switch ($choice.Trim()) {
        "1" { return "tailscale" }
        "2" { return "cloudflare" }
        "3" { return "ngrok" }
        "4" { return "custom" }
        default { return "tailscale" }
    }
}

function Start-NexusTunnel {
    param([string]$Provider)

    $localUrl = "http://127.0.0.1:$Port"
    if ($Provider -eq "tailscale") {
        $cmd = Get-Command tailscale -ErrorAction SilentlyContinue
        if (!$cmd) {
            throw "Tailscale CLI not found. Install Tailscale and enable Funnel, or choose another provider."
        }
        Write-NexusLine "Starting Tailscale Funnel for $localUrl ..." "Ok"
        & $cmd.Source funnel $Port
        return
    }
    if ($Provider -eq "cloudflare") {
        $cmd = Get-Command cloudflared -ErrorAction SilentlyContinue
        if (!$cmd) {
            throw "cloudflared not found. Install Cloudflare Tunnel client or choose another provider."
        }
        Write-NexusLine "Starting Cloudflare Quick Tunnel for $localUrl ..." "Warn"
        & $cmd.Source tunnel --url $localUrl
        return
    }
    if ($Provider -eq "ngrok") {
        $cmd = Get-Command ngrok -ErrorAction SilentlyContinue
        if (!$cmd) {
            throw "ngrok not found. Install/configure ngrok or choose another provider."
        }
        Write-NexusLine "Starting ngrok HTTP tunnel for $localUrl ..." "Warn"
        & $cmd.Source http $Port
        return
    }
    if ([string]::IsNullOrWhiteSpace($env:NEXUS_ONLINE_TUNNEL_COMMAND)) {
        throw "NEXUS_ONLINE_TUNNEL_COMMAND is empty. Example: set it to your own FRP/WireGuard/reverse-proxy command and use {port}."
    }
    $command = $env:NEXUS_ONLINE_TUNNEL_COMMAND.Replace("{port}", [string]$Port)
    Write-NexusLine "Starting custom tunnel command..." "Info"
    & cmd /c $command
}

function Test-NexusTunnelProvider {
    param([string]$Provider)

    if ($Provider -eq "tailscale") {
        return [bool](Get-Command tailscale -ErrorAction SilentlyContinue)
    }
    if ($Provider -eq "cloudflare") {
        return [bool](Get-Command cloudflared -ErrorAction SilentlyContinue)
    }
    if ($Provider -eq "ngrok") {
        return [bool](Get-Command ngrok -ErrorAction SilentlyContinue)
    }
    return ![string]::IsNullOrWhiteSpace($env:NEXUS_ONLINE_TUNNEL_COMMAND)
}

function Write-NexusTunnelInstallHint {
    param([string]$Provider)

    if ($Provider -eq "tailscale") {
        Write-NexusLine "Tailscale CLI was not found. Install Tailscale for Windows, log in, enable Funnel, then choose Tailscale again." "Warn"
        Write-NexusLine "Install: https://tailscale.com/download/windows" "Info"
        return
    }
    if ($Provider -eq "cloudflare") {
        Write-NexusLine "cloudflared was not found. Install Cloudflare Tunnel client or choose another provider." "Warn"
        Write-NexusLine "Install: https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/" "Info"
        return
    }
    if ($Provider -eq "ngrok") {
        Write-NexusLine "ngrok was not found. Install ngrok and configure your auth token, or choose another provider." "Warn"
        Write-NexusLine "Install: https://ngrok.com/download" "Info"
        return
    }
    Write-NexusLine "NEXUS_ONLINE_TUNNEL_COMMAND is empty. Set it with a command that forwards {port}, or choose another provider." "Warn"
}

function Install-NexusTunnelProvider {
    param([string]$Provider)

    if ($Provider -eq "custom") {
        Write-NexusLine "Custom tunnel commands cannot be installed automatically." "Warn"
        return $false
    }

    $winget = Get-Command winget -ErrorAction SilentlyContinue
    if (!$winget) {
        Write-NexusLine "winget was not found on this Windows install. Use the manual install link above, then run StartTunnel.bat again." "Warn"
        return $false
    }

    $packageId = switch ($Provider) {
        "tailscale" { "Tailscale.Tailscale" }
        "cloudflare" { "Cloudflare.cloudflared" }
        "ngrok" { "Ngrok.Ngrok" }
        default { "" }
    }
    if ([string]::IsNullOrWhiteSpace($packageId)) {
        return $false
    }

    Write-NexusLine "Installing optional tunnel dependency with winget: $packageId" "Info"
    & $winget.Source install --id $packageId --exact --accept-package-agreements --accept-source-agreements
    if ($LASTEXITCODE -ne 0) {
        Write-NexusLine "winget install failed or was cancelled." "Warn"
        return $false
    }
    return $true
}

function Resolve-NexusTunnelProvider {
    $provider = Select-NexusTunnelProvider
    while (!(Test-NexusTunnelProvider -Provider $provider)) {
        Write-NexusTunnelInstallHint -Provider $provider
        if ($provider -ne "custom") {
            $install = Read-Host "Install this optional tunnel dependency now? [y/N]"
            if ($install.Trim().ToLowerInvariant() -eq "y") {
                [void](Install-NexusTunnelProvider -Provider $provider)
                if (Test-NexusTunnelProvider -Provider $provider) {
                    return $provider
                }
                Write-NexusLine "The command is still not available in this terminal. If installation completed, reopen StartTunnel.bat." "Warn"
            }
        }
        Write-NexusLine "Nexus is still running locally at http://127.0.0.1:$Port/ui." "Info"
        $answer = Read-Host "Choose another tunnel provider? [Y/n]"
        if ($answer.Trim().ToLowerInvariant() -eq "n") {
            return ""
        }
        $script:TunnelProvider = "ask"
        $provider = Select-NexusTunnelProvider
    }
    return $provider
}

Write-NexusLogo
Write-NexusSection $(if ($Mode -eq "lan") { "LAN startup" } else { "Tunnel startup" })
Write-NexusLine "Preparing local model/path configuration..." "Info"
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\configure_model_paths.ps1") -ProjectRoot $root
if ($LASTEXITCODE -ne 0) {
    throw "Model path setup failed."
}
Import-NexusStartupEnv

$env:NEXUS_BACKEND_HOST = if ($Mode -eq "lan") { "0.0.0.0" } else { "127.0.0.1" }
$env:NEXUS_BACKEND_PORT = [string]$Port

$startArgs = @(
    "-NoProfile",
    "-ExecutionPolicy",
    "Bypass",
    "-File",
    (Join-Path $root "scripts\start_nexus.ps1"),
    "-ProjectRoot",
    $root,
    "-ComfyWarmupSeconds",
    "0",
    "-NoOpen"
)
if (!$SkipComfy) {
    $startArgs += "-StartComfy"
}

Write-NexusLine "Starting Nexus at http://127.0.0.1:$Port/ui ..." "Info"
& powershell @startArgs
if ($LASTEXITCODE -ne 0) {
    throw "Nexus startup failed."
}

try {
    if ($Mode -eq "lan") {
        Write-NexusSection "LAN access"
        Write-NexusLine "No external tunnel is running. Use only your local network/VPN." "Ok"
        Write-NexusLine "Local: http://127.0.0.1:$Port/ui" "Info"
        $lanUrls = @(Get-NexusLanUrls)
        if ($lanUrls.Count -gt 0) {
            foreach ($url in $lanUrls) {
                Write-NexusLine "LAN: $url" "Info"
            }
        } else {
            Write-NexusLine "No LAN IPv4 address was detected. Check Windows network/firewall settings." "Warn"
        }
        Write-NexusLine "Press any key to close and stop services." "Info"
        [void][System.Console]::ReadKey($true)
    } else {
        $provider = Resolve-NexusTunnelProvider
        if ([string]::IsNullOrWhiteSpace($provider)) {
            Write-NexusLine "Tunnel not started. Press any key to close and stop services." "Info"
            [void][System.Console]::ReadKey($true)
            return
        }
        Write-NexusSection "Tunnel"
        Write-NexusLine "Nexus is local at http://127.0.0.1:$Port/ui. The selected tunnel will forward this local URL." "Info"
        Write-NexusLine "Only share the public URL with trusted users; add authentication before public use." "Warn"
        Start-NexusTunnel -Provider $provider
    }
} finally {
    Write-NexusLine "Stopping Nexus services..." "Info"
    & powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $root "scripts\stop_nexus.ps1") -ProjectRoot $root
}
