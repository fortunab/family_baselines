# Setup script using Standard Python venv for Windows
$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   PYTHON VENV ENVIRONMENT SETUP (SKORCH)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$venvName = "venv_cervical_skorch"

if (-not (Test-Path $venvName)) {
    Write-Host "Creating Python virtual environment in $venvName..." -ForegroundColor Cyan
    python -m venv $venvName
}

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\$venvName\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing requirements.txt..." -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Python venv ready in $venvName!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "To run fast dry-run: python main_federated_skorch.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb" -ForegroundColor Yellow
Write-Host "To run training:     python main_federated_skorch.py --config configs/default.toml" -ForegroundColor Yellow
Write-Host "To run linting:      python run_linter.py" -ForegroundColor Yellow
