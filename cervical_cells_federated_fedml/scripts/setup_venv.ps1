# Standard venv Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Python venv for FedML Cervical Cytology        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$VENV_DIR = "venv_federated_fedml"

Write-Host "[*] Creating standard virtual environment at '$VENV_DIR'..." -ForegroundColor Green
python -m venv $VENV_DIR

Write-Host "[*] Activating virtual environment..." -ForegroundColor Green
& ".\$VENV_DIR\Scripts\Activate.ps1"

Write-Host "[*] Upgrading pip, setuptools, and wheel..." -ForegroundColor Green
python -m pip install --upgrade pip setuptools wheel

Write-Host "[*] Installing dependencies from requirements.txt..." -ForegroundColor Green
pip install -r requirements.txt

Write-Host "`n[SUCCESS] Virtual environment '$VENV_DIR' is ready!" -ForegroundColor Green
Write-Host "To activate:   .\$VENV_DIR\Scripts\Activate.ps1"
Write-Host "Run fastai:    python main_federated_fedml.py --framework fastai"
Write-Host "Run skorch:    python main_federated_fedml.py --framework skorch"
