# ==============================================================================
# Pixi Automated Setup Script for Windows PowerShell
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   SUBSTRA CERVICAL FL: PIXI ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Locate or install Pixi
$pixiCmd = Get-Command pixi -ErrorAction SilentlyContinue
if (-not $pixiCmd) {
    $fallbackPath = "$env:USERPROFILE\.pixi\bin\pixi.exe"
    if (Test-Path $fallbackPath) {
        $env:PATH = "$env:USERPROFILE\.pixi\bin;$env:PATH"
        Write-Host "[Pixi-Bootstrap] Found Pixi at $fallbackPath. Added to PATH." -ForegroundColor Green
    } else {
        Write-Host "[Pixi-Bootstrap] Pixi not detected. Installing official Pixi binary..." -ForegroundColor Yellow
        try {
            Invoke-RestMethod -Uri https://pixi.sh/install.ps1 | Invoke-Expression
            $env:PATH = "$env:USERPROFILE\.pixi\bin;$env:PATH"
            Write-Host "[Pixi-Bootstrap] Pixi successfully installed!" -ForegroundColor Green
        } catch {
            Write-Error "Failed to install Pixi automatically. Please install from https://pixi.sh"
            exit 1
        }
    }
}

# 2. Synchronize environment & lockfile
Write-Host "[Pixi-Setup] Solving and installing dependencies via pixi.toml..." -ForegroundColor Cyan
pixi install

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "   PIXI SETUP COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To execute tasks:" -ForegroundColor Yellow
Write-Host "  pixi run train-fastai   # Run Substra training with fastai" -ForegroundColor White
Write-Host "  pixi run train-skorch   # Run Substra training with skorch" -ForegroundColor White
Write-Host "  pixi run dry-run        # Fast dry-run with Random Best Select" -ForegroundColor White
Write-Host "  pixi run benchmark      # Compare strategies & frameworks" -ForegroundColor White
Write-Host "  pixi run lint           # Static code quality checker" -ForegroundColor White
