param(
    [string]$ProjectRoot = "D:\NexusBTA",
    [string]$ModelsSource = "C:\ComfyUpdate\models",
    [switch]$IncludeErosModel,
    [switch]$IncludeSulphurModel,
    [switch]$IncludeAllLtxLoras,
    [switch]$Force
)

$ErrorActionPreference = "Stop"

function Copy-FileAsset([string]$RelativePath) {
    $source = Join-Path $ModelsSource $RelativePath
    $target = Join-Path (Join-Path $ProjectRoot "models") $RelativePath
    if (!(Test-Path -LiteralPath $source)) {
        Write-Warning "Missing source asset: $source"
        return
    }
    $targetDir = Split-Path -Parent $target
    New-Item -ItemType Directory -Path $targetDir -Force | Out-Null
    if ((Test-Path -LiteralPath $target) -and !$Force) {
        Write-Host "Exists: $RelativePath"
        return
    }
    Copy-Item -LiteralPath $source -Destination $target -Force:$Force
    Write-Host "Copied: $RelativePath"
}

$assets = @(
    "unet\ltx-2.3-22b-dev-Q4_K_M.gguf",
    "text_encoders\gemma-3-12b-it-heretic-v2_nvfp4.safetensors",
    "text_encoders\ltx-2.3_text_projection_bf16.safetensors",
    "vae\LTX23_video_vae_bf16.safetensors",
    "vae\LTX23_audio_vae_bf16.safetensors",
    "vae\taeltx2_3.safetensors",
    "latent_upscale_models\ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
    "loras\LTX2\ltx-2.3-22b-distilled-lora-1.1_fro90_ceil72_condsafe.safetensors",
    "loras\LTX2\ltx-2.3-22b-distilled-lora-384-1.1.safetensors",
    "loras\LTX2\LTX2.3-22B_IC-LoRA-Cameraman_v1_10500.safetensors",
    "loras\LTX2\LTX2.3_reasoning_I2V_V3.safetensors"
)

if ($IncludeErosModel) {
    $assets += "diffusion_models\10Eros_v1-fp8mixed_learned.safetensors"
}

if ($IncludeSulphurModel) {
    $assets += "diffusion_models\sulphur_dev_fp8mixed.safetensors"
}

foreach ($asset in $assets) {
    Copy-FileAsset $asset
}

if ($IncludeAllLtxLoras) {
    $sourceLoras = Join-Path $ModelsSource "loras\LTX2"
    $targetLoras = Join-Path $ProjectRoot "models\loras\LTX2"
    New-Item -ItemType Directory -Path $targetLoras -Force | Out-Null
    Get-ChildItem -LiteralPath $sourceLoras -File | ForEach-Object {
        $target = Join-Path $targetLoras $_.Name
        if (!(Test-Path -LiteralPath $target) -or $Force) {
            Copy-Item -LiteralPath $_.FullName -Destination $target -Force:$Force
            Write-Host "Copied: loras\LTX2\$($_.Name)"
        }
    }
}

Write-Host "LTX 2.3 asset import completed."
