param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$WorkflowsDir = "",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)

if ([string]::IsNullOrWhiteSpace($WorkflowsDir)) {
    $settingsPath = Join-Path $root "config\nexus_settings.json"
    if (Test-Path -LiteralPath $settingsPath) {
        $settings = Get-Content -LiteralPath $settingsPath -Raw | ConvertFrom-Json
        $WorkflowsDir = [string]$settings.workflows_dir
    }
}
if ([string]::IsNullOrWhiteSpace($WorkflowsDir)) { $WorkflowsDir = Join-Path $root "workflows\comfyui" }

New-Item -ItemType Directory -Force -Path $WorkflowsDir | Out-Null
$workflows = @{
    "minimax_h3_t2v.json" = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_t2v.json"
    "minimax_h3_i2v.json" = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_i2v.json"
    "minimax_h3_r2v.json" = "https://raw.githubusercontent.com/Comfy-Org/workflow_templates/main/templates/video_minimax_h3_r2v.json"
}

foreach ($entry in $workflows.GetEnumerator()) {
    $target = Join-Path $WorkflowsDir $entry.Key
    if ((Test-Path -LiteralPath $target) -and !$Force) {
        Write-Host "[NEXUS BTA] Preserving existing MiniMax H3 workflow: $($entry.Key)"
        continue
    }
    Invoke-WebRequest -Uri $entry.Value -OutFile $target
    Write-Host "[NEXUS BTA] Imported official MiniMax H3 workflow: $($entry.Key)"
}
