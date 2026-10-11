# Standard venv Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Python venv for FedML Cytology                 " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

Write-Host "[*] Creating standard virtual environment at .venv..." -ForegroundColor Green
python -m venv .venv

Write-Host "[*] Upgrading pip and installing requirements..." -ForegroundColor Green
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`n[SUCCESS] Standard venv environment configured!" -ForegroundColor Green
Write-Host "To activate:        .\.venv\Scripts\Activate.ps1"
Write-Host "To execute FedML:   python main_federated_fedml.py --preprocessing n2n_cyclegan"
