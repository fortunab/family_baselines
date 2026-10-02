# Conda Environment Setup Script for Windows PowerShell
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Conda Environment for FedML Cervical Cytology  " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ENV_NAME = "cervical_cells_fedml"

Write-Host "[*] Creating Conda environment from environment.yml..." -ForegroundColor Green
conda env create -f environment.yml --overwrite

Write-Host "`n[SUCCESS] Conda environment '$ENV_NAME' created successfully!" -ForegroundColor Green
Write-Host "To activate:   conda activate $ENV_NAME"
Write-Host "Run fastai:    python main_federated_fedml.py --framework fastai"
Write-Host "Run skorch:    python main_federated_fedml.py --framework skorch"
