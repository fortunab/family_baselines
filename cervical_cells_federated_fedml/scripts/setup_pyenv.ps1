# pyenv-win Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Pyenv Environment for FedML Cervical Cytology  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$TARGET_PYTHON = "3.11.9"

if (-not (Get-Command "pyenv" -ErrorAction SilentlyContinue)) {
    Write-Host "[!] pyenv-win not found. Please install via: Invoke-WebRequest -UseBasicParsing -Uri 'https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1' | Invoke-Expression" -ForegroundColor Yellow
    exit 1
}

Write-Host "[*] Installing Python $TARGET_PYTHON via pyenv..." -ForegroundColor Green
pyenv install -s $TARGET_PYTHON

Write-Host "[*] Setting local directory Python version to $TARGET_PYTHON..." -ForegroundColor Green
pyenv local $TARGET_PYTHON

Write-Host "[*] Initializing dedicated virtualenv..." -ForegroundColor Green
python -m venv venv_pyenv_fedml
& ".\venv_pyenv_fedml\Scripts\Activate.ps1"
pip install -r requirements.txt

Write-Host "`n[SUCCESS] Pyenv environment setup complete!" -ForegroundColor Green
Write-Host "Run fastai:    python main_federated_fedml.py --framework fastai"
Write-Host "Run skorch:    python main_federated_fedml.py --framework skorch"
