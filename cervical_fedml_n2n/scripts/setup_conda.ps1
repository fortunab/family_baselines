# Conda Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Conda Environment for FedML Cytology          " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$EnvName = "cervical_cells_fedml_n2n_cyclegan"

Write-Host "[*] Creating Conda environment from environment.yml..." -ForegroundColor Green
conda env create -f environment.yml --name $EnvName --yes

Write-Host "`n[SUCCESS] Conda environment '$EnvName' created!" -ForegroundColor Green
Write-Host "To activate:        conda activate $EnvName"
Write-Host "To execute FedML:   python main_federated_fedml.py --preprocessing n2n_cyclegan"
