# Setup Python Virtual Environment on Windows PowerShell
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up venv for Dual-Database DR & ISIC/HAM10000 ML " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VENV_PATH = ".\venv"

if (-Not (Test-Path $VENV_PATH)) {
    Write-Host "[1/4] Creating virtual environment at $VENV_PATH..." -ForegroundColor Yellow
    python -m venv $VENV_PATH
} else {
    Write-Host "[1/4] Virtual environment already exists at $VENV_PATH." -ForegroundColor Green
}

Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& "$VENV_PATH\Scripts\Activate.ps1"

Write-Host "[3/4] Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
& "$VENV_PATH\Scripts\python.exe" -m pip install --upgrade pip setuptools wheel

Write-Host "[4/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
& "$VENV_PATH\Scripts\python.exe" -m pip install -r requirements.txt

Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Setup Complete! Activate with: .\venv\Scripts\Activate.ps1" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
