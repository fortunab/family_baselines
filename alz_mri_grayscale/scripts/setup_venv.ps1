# Setup script for Python venv on Windows (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "Creating Python virtual environment in .\venv..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

Write-Host "Upgrading pip and installing requirements..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "Setup complete! Virtual environment is ready." -ForegroundColor Green
