# Conda Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Conda for Flower & Substra VAE Cytology        " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$EnvName = "cervical_cells_vae_flower_substra"

Write-Host "[*] Creating Conda environment from environment.yml..." -ForegroundColor Green
conda env create -f environment.yml --name $EnvName --yes

Write-Host "`n[SUCCESS] Conda environment '$EnvName' created!" -ForegroundColor Green
Write-Host "To activate:        conda activate $EnvName"
Write-Host "To execute Flower:  python main_federated_flower.py --preprocessing vae"
Write-Host "To execute Substra: python main_federated_substra.py --preprocessing vae"
