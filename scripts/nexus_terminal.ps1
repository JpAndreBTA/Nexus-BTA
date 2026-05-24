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
        Write-NexusLine "Git repository not found; skipping updates." "Warn"
        return
    }
    if (!(Get-Command git -ErrorAction SilentlyContinue)) {
        Write-NexusLine "Git was not found in PATH; skipping updates." "Warn"
        return
    }

    try {
        Write-NexusLine "Checking GitHub..." "Info"
        git -C $ProjectRoot fetch --all --prune 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "git fetch failed"
        }

        $dirty = git -C $ProjectRoot status --porcelain
        if ($dirty) {
            Write-NexusLine "Local changes detected; automatic pull skipped." "Warn"
            return
        }

        $upstream = git -C $ProjectRoot rev-parse --abbrev-ref --symbolic-full-name "@{u}" 2>$null
        if ($LASTEXITCODE -ne 0 -or !$upstream) {
            Write-NexusLine "No upstream branch configured; automatic pull skipped." "Warn"
            return
        }

        $counts = (git -C $ProjectRoot rev-list --left-right --count "$upstream...HEAD").Trim() -split "\s+"
        $behind = if ($counts.Length -gt 0) { [int]$counts[0] } else { 0 }
        $ahead = if ($counts.Length -gt 1) { [int]$counts[1] } else { 0 }

        if ($behind -eq 0 -and $ahead -eq 0) {
            Write-NexusLine "Repository is up to date." "Ok"
            return
        }
        if ($ahead -gt 0 -and $behind -gt 0) {
            Write-NexusLine "Branch diverged from remote; resolve it manually before pulling." "Warn"
            return
        }
        if ($ahead -gt 0) {
            Write-NexusLine "Local commits are not pushed yet; automatic pull skipped." "Warn"
            return
        }

        Write-NexusLine "Applying $behind GitHub update(s)..." "Info"
        git -C $ProjectRoot pull --ff-only 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "git pull --ff-only failed"
        }
        Write-NexusLine "Update applied." "Ok"
    } catch {
        if ($Strict) {
            throw
        }
        Write-NexusLine "Could not check for updates: $($_.Exception.Message)" "Warn"
    }
}
