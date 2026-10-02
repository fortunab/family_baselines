# ==============================================================================
# Standard Python venv Automated Setup Script for Windows PowerShell
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   SUBSTRA CERVICAL FL: PYTHON VENV SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Create virtual environment
Write-Host "[venv-Setup] Creating virtual environment in .venv..." -ForegroundColor Cyan
python -m venv .venv

# 2. Upgrade pip and wheel
Write-Host "[venv-Setup] Upgrading pip and wheel inside .venv..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install --upgrade pip wheel setuptools

# 3. Install requirements
Write-Host "[venv-Setup] Installing dependencies from requirements.txt..." -ForegroundColor Cyan
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "   VENV SETUP COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To activate and run:" -ForegroundColor Yellow
Write-Host "  .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "  python main_federated_substra.py --config configs/default.toml --framework fastai" -ForegroundColor White
Write-Host "  python main_federated_substra.py --config configs/default.toml --framework skorch" -ForegroundColor White
Write-Host "  python compare_substra_strategies.py --rounds 3" -ForegroundColor White
Write-Host "  python run_linter.py" -ForegroundColor White
