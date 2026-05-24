$script:NexusEsc = [char]27

function Get-NexusBrand {
    "$($script:NexusEsc)[91mNEXUS$($script:NexusEsc)[0m $($script:NexusEsc)[96mBTA$($script:NexusEsc)[0m"
}

function Write-NexusLogo {
    $brand = Get-NexusBrand
    Write-Host ""
    Write-Host "$brand $($script:NexusEsc)[90mLOCAL GENERATION STUDIO$($script:NexusEsc)[0m"
    Write-Host "$($script:NexusEsc)[90m----------------------------------------$($script:NexusEsc)[0m"
}

function Write-NexusLine {
    param(
        [string]$Message,
        [ValidateSet("Info", "Ok", "Warn", "Error", "Step")]
        [string]$Kind = "Info"
    )

    $brand = Get-NexusBrand
    $color = switch ($Kind) {
        "Ok" { "$($script:NexusEsc)[92m" }
        "Warn" { "$($script:NexusEsc)[93m" }
        "Error" { "$($script:NexusEsc)[91m" }
        "Step" { "$($script:NexusEsc)[95m" }
        default { "$($script:NexusEsc)[90m" }
    }
    $label = switch ($Kind) {
        "Ok" { "OK" }
        "Warn" { "WARN" }
        "Error" { "FAIL" }
        "Step" { "STEP" }
        default { "INFO" }
    }

    Write-Host "$brand $color$($label.PadRight(4))$($script:NexusEsc)[0m $Message"
}

function Write-NexusSection {
    param([string]$Title)
    Write-Host ""
    Write-NexusLine $Title "Step"
}

function Invoke-NexusRepositoryUpdate {
    param(
        [string]$ProjectRoot,
        [switch]$Strict
    )

    if (!(Test-Path -LiteralPath (Join-Path $ProjectRoot ".git"))) {
        Write-NexusLine "Repositorio Git nao encontrado; pulando atualizacoes." "Warn"
        return
    }
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        Write-NexusLine "Git nao encontrado no PATH; pulando atualizacoes." "Warn"
        return
    }

    try {
        Write-NexusLine "Verificando GitHub..." "Info"
        git -C $ProjectRoot fetch --all --prune 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "git fetch falhou"
        }

        $dirty = git -C $ProjectRoot status --porcelain
        if ($dirty) {
            Write-NexusLine "Alteracoes locais detectadas; pull automatico pulado." "Warn"
            return
        }

        $upstream = git -C $ProjectRoot rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
        if ($LASTEXITCODE -ne 0 -or !$upstream) {
            Write-NexusLine "Branch sem upstream configurado; pull automatico pulado." "Warn"
            return
        }

        $counts = (git -C $ProjectRoot rev-list --left-right --count "$upstream...HEAD").Trim() -split "\s+"
        $behind = if ($counts.Length -gt 0) { [int]$counts[0] } else { 0 }
        $ahead = if ($counts.Length -gt 1) { [int]$counts[1] } else { 0 }

        if ($behind -eq 0 -and $ahead -eq 0) {
            Write-NexusLine "Repositorio atualizado." "Ok"
            return
        }
        if ($ahead -gt 0 -and $behind -gt 0) {
            Write-NexusLine "Branch divergiu do remoto; resolva manualmente antes do pull." "Warn"
            return
        }
        if ($ahead -gt 0) {
            Write-NexusLine "Commits locais ainda nao enviados; pull automatico pulado." "Warn"
            return
        }

        Write-NexusLine "Aplicando $behind atualizacao(oes) do GitHub..." "Info"
        git -C $ProjectRoot pull --ff-only 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "git pull --ff-only falhou"
        }
        Write-NexusLine "Atualizacao aplicada." "Ok"
    } catch {
        if ($Strict) {
            throw
        }
        Write-NexusLine "Nao foi possivel verificar atualizacoes: $($_.Exception.Message)" "Warn"
    }
}
