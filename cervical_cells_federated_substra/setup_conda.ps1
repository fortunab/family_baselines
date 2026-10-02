# ==============================================================================
# Conda / Mamba Automated Setup Script for Windows PowerShell
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

Write-Host "======================================================================" -ForegroundColor Cyan
Write-Host "   SUBSTRA CERVICAL FL: CONDA ENVIRONMENT SETUP" -ForegroundColor Cyan
Write-Host "======================================================================" -ForegroundColor Cyan

# 1. Check for conda or mamba
$condaCmd = Get-Command conda -ErrorAction SilentlyContinue
if (-not $condaCmd) {
    Write-Error "Conda not found in PATH. Please launch Anaconda Prompt or install Miniconda."
    exit 1
}

# 2. Create or update conda environment from environment.yml
Write-Host "[Conda-Setup] Creating conda environment 'cervical_fl_substra'..." -ForegroundColor Cyan
conda env create -f environment.yml --overwrite

Write-Host "`n======================================================================" -ForegroundColor Green
Write-Host "   CONDA SETUP COMPLETE!" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "To activate and run:" -ForegroundColor Yellow
Write-Host "  conda activate cervical_fl_substra" -ForegroundColor White
Write-Host "  python main_federated_substra.py --config configs/default.toml --framework fastai" -ForegroundColor White
Write-Host "  python main_federated_substra.py --config configs/default.toml --framework skorch" -ForegroundColor White
Write-Host "  python compare_substra_strategies.py --rounds 3" -ForegroundColor White
Write-Host "  python run_linter.py" -ForegroundColor White
