# Setup Conda Environment on Windows PowerShell
Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Setting up Conda for Dual-Database DR & ISIC/HAM10000 ML " -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

$ENV_NAME = "dr_dual_db_env"

Write-Host "[1/2] Creating/Updating environment from environment.yml..." -ForegroundColor Yellow
conda env create -f environment.yml -n $ENV_NAME --yes 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Updating existing environment..." -ForegroundColor Yellow
    conda env update -f environment.yml -n $ENV_NAME --prune
}

Write-Host "[2/2] Conda setup finished." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host " Activate environment with: conda activate $ENV_NAME" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
