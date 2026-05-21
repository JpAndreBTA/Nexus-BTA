param(
    [string]$ProjectRoot = "D:\NexusBTA"
)

$ErrorActionPreference = "Stop"

$root = $executionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($ProjectRoot)
$qwenNodes = Join-Path $root "runtime\ComfyUI\comfy_extras\nodes_qwen.py"

if (Test-Path -LiteralPath $qwenNodes) {
    $content = Get-Content -LiteralPath $qwenNodes -Raw
    if ($content -match "node_helpers\.conditioning_set_values") {
        $patched = $content -replace "(?m)^import node_helpers\r?\n", ""
        if ($patched -match "(?m)^import comfy\.model_management\r?$") {
            $patched = $patched -replace "(?m)^(import comfy\.model_management\r?\n)", "`$1import node_helpers`r`n"
        } else {
            $patched = "import node_helpers`r`n$patched"
        }
        if ($patched -ne $content) {
            Set-Content -LiteralPath $qwenNodes -Value $patched -Encoding UTF8
            Write-Host "[NEXUS BTA] Applied ComfyUI Qwen node_helpers hotfix."
        }
    }
}
