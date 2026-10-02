# uv Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up uv Environment for FedML Cervical Cytology     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "[*] uv not found. Installing via Astral official installer..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
}

Write-Host "[*] Creating virtual environment (.venv) using Python 3.11..." -ForegroundColor Green
uv venv --python 3.11 .venv

Write-Host "[*] Installing project dependencies into .venv..." -ForegroundColor Green
uv pip install -e ".[dev]"

Write-Host "`n[SUCCESS] uv environment is ready!" -ForegroundColor Green
Write-Host "Activate with: .\.venv\Scripts\Activate.ps1"
Write-Host "Run fastai:    python main_federated_fedml.py --framework fastai"
Write-Host "Run skorch:    python main_federated_fedml.py --framework skorch"
