# Setup script using Conda / Mamba for Windows
Write-Host "Creating Conda environment 'cervical_flower_env' from environment.yml..." -ForegroundColor Cyan
conda env create -f environment.yml
Write-Host "`n[SUCCESS] Conda environment created!" -ForegroundColor Green
Write-Host "To activate: conda activate cervical_flower_env" -ForegroundColor Yellow
Write-Host "To run:      python main_federated_flower.py --config configs/default.toml" -ForegroundColor Yellow
