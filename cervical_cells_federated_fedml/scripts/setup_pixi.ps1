# Pixi Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Pixi Environment for FedML Cervical Cytology   " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

if (-not (Get-Command "pixi" -ErrorAction SilentlyContinue)) {
    Write-Host "[*] Pixi not found. Installing via official installer..." -ForegroundColor Yellow
    iwr -useb https://pixi.sh/install.ps1 | iex
    $env:Path += ";$env:USERPROFILE\.pixi\bin"
}

Write-Host "[*] Resolving and installing Pixi dependencies..." -ForegroundColor Green
pixi install

Write-Host "[*] Verifying environment by running static quality checks..." -ForegroundColor Green
pixi run lint

Write-Host "`n[SUCCESS] Pixi environment is ready!" -ForegroundColor Green
Write-Host "To execute FedML with fastai:  pixi run python main_federated_fedml.py --framework fastai"
Write-Host "To execute FedML with skorch:  pixi run python main_federated_fedml.py --framework skorch"
