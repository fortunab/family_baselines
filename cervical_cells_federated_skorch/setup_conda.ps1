# Setup script using Conda / Mamba for Windows
$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   CONDA / MAMBA ENVIRONMENT SETUP (SKORCH)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$envName = "cervical_cells_skorch"

if (Get-Command mamba -ErrorAction SilentlyContinue) {
    Write-Host "Creating environment using Mamba..." -ForegroundColor Cyan
    mamba env create -f environment.yml
} elseif (Get-Command conda -ErrorAction SilentlyContinue) {
    Write-Host "Creating environment using Conda..." -ForegroundColor Cyan
    conda env create -f environment.yml
} else {
    Write-Error "Neither Conda nor Mamba was found in PATH."
}

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Conda environment '$envName' created!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "Activate with: conda activate $envName" -ForegroundColor Yellow
Write-Host "Then run:      python main_federated_skorch.py --config configs/default.toml" -ForegroundColor Yellow
