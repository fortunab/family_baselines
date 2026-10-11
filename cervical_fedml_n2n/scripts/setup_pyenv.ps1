# Pyenv Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Pyenv for FedML Cytology                       " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$TargetVersion = Get-Content .python-version -Raw
$TargetVersion = $TargetVersion.Trim()

Write-Host "[*] Pyenv target version: $TargetVersion" -ForegroundColor Yellow

if (Get-Command "pyenv" -ErrorAction SilentlyContinue) {
    Write-Host "[*] Installing Python $TargetVersion via pyenv-win..." -ForegroundColor Green
    pyenv install --skip-existing $TargetVersion
    pyenv local $TargetVersion
} else {
    Write-Host "[!] pyenv not found in system path. Please ensure Python $TargetVersion is installed." -ForegroundColor Yellow
}

Write-Host "[*] Creating virtual environment with active Python..." -ForegroundColor Green
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements.txt

Write-Host "`n[SUCCESS] Pyenv environment setup complete!" -ForegroundColor Green
Write-Host "To activate: .\.venv\Scripts\Activate.ps1"
