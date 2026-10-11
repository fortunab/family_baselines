# uv Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up uv Virtual Environment for FedML Cytology     " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Get-Command "uv" -ErrorAction SilentlyContinue)) {
    Write-Host "[*] uv not found. Installing via official installer..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path += ";$env:USERPROFILE\.cargo\bin"
}

Write-Host "[*] Creating virtual environment with Python 3.11..." -ForegroundColor Green
uv venv --python 3.11 .venv

Write-Host "[*] Synchronizing dependencies using uv pip..." -ForegroundColor Green
uv pip install -r requirements.txt

Write-Host "`n[SUCCESS] uv environment configured at .venv!" -ForegroundColor Green
Write-Host "To activate:        .\.venv\Scripts\Activate.ps1"
Write-Host "To execute FedML:   uv run python main_federated_fedml.py --preprocessing n2n_cyclegan"
