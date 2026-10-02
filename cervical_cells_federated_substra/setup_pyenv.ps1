# ==============================================================================
# Pyenv Automated Setup Script for Windows PowerShell
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   SUBSTRA CERVICAL FL: PYENV ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Verify pyenv
$pyenvCmd = Get-Command pyenv -ErrorAction SilentlyContinue
if (-not $pyenvCmd) {
    Write-Warning "Pyenv for Windows (pyenv-win) not detected in PATH."
    Write-Host "If installed, ensure ~/.pyenv/pyenv-win/bin is in your PATH." -ForegroundColor Yellow
} else {
    Write-Host "[Pyenv-Setup] Pinned Python version (.python-version): 3.11.9" -ForegroundColor Cyan
    pyenv install 3.11.9 --skip-existing
    pyenv local 3.11.9
}

# 2. Virtual environment using pinned Python
Write-Host "[Pyenv-Setup] Setting up virtual environment with Python 3.11.9..." -ForegroundColor Cyan
python -m venv .venv_pyenv
& .\.venv_pyenv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv_pyenv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "   PYENV SETUP COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To activate and run:" -ForegroundColor Yellow
Write-Host "  .\.venv_pyenv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python main_federated_substra.py --config configs/default.toml --framework fastai" -ForegroundColor White
Write-Host "  python main_federated_substra.py --config configs/default.toml --framework skorch" -ForegroundColor White
