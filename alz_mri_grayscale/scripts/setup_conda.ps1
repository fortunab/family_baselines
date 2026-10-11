# Setup script for Conda on Windows (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "Creating Conda environment 'alzheimer-grayscale-bench' from environment.yml..." -ForegroundColor Cyan
conda env create -f environment.yml

Write-Host "Conda environment created successfully. Activate with: conda activate alzheimer-grayscale-bench" -ForegroundColor Green
