# ==============================================================================
# uv Automated Setup Script for Windows PowerShell
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   SUBSTRA CERVICAL FL: UV ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Locate or install uv
$uvCmd = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uvCmd) {
    $fallbackUv = "$env:USERPROFILE\.cargo\bin\uv.exe"
    if (Test-Path $fallbackUv) {
        $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    } else {
        Write-Host "[uv-Bootstrap] Installing standalone uv binary..." -ForegroundColor Yellow
        powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
        $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
    }
}

# 2. Create virtual environment using uv
Write-Host "[uv-Setup] Creating virtual environment (.venv) using Python 3.11..." -ForegroundColor Cyan
uv venv .venv --python 3.11

# 3. Synchronize project dependencies from pyproject.toml
Write-Host "[uv-Setup] Installing dependencies via uv pip..." -ForegroundColor Cyan
uv pip install -e .

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "   UV SETUP COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To execute tasks:" -ForegroundColor Yellow
Write-Host "  uv run python main_federated_substra.py --config configs/default.toml --framework fastai" -ForegroundColor White
Write-Host "  uv run python main_federated_substra.py --config configs/default.toml --framework skorch" -ForegroundColor White
Write-Host "  uv run python compare_substra_strategies.py --rounds 3" -ForegroundColor White
Write-Host "  uv run python run_linter.py" -ForegroundColor White
