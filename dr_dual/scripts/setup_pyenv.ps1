# Setup Pyenv Environment on Windows PowerShell
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Pyenv for Dual-Database DR & ISIC/HAM10000 ML " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$PYTHON_VERSION = "3.11.9"

Write-Host "[1/3] Ensuring Python $PYTHON_VERSION is installed via pyenv-win..." -ForegroundColor Yellow
pyenv install -s $PYTHON_VERSION

Write-Host "[2/3] Setting local Python version..." -ForegroundColor Yellow
pyenv local $PYTHON_VERSION

Write-Host "[3/3] Creating and activating pyenv virtual environment..." -ForegroundColor Yellow
python -m venv ".\.venv"
& ".\.venv\Scripts\Activate.ps1"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r requirements.txt

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Pyenv setup complete! Activate with: .\.venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
