# Setup script using standard Python venv
Write-Host "Creating Python venv environment 'venv_flower_cervical'..." -ForegroundColor Cyan
python -m venv venv_flower_cervical
Write-Host "Activating environment..." -ForegroundColor Cyan
.\venv_flower_cervical\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
Write-Host "Installing requirements..." -ForegroundColor Cyan
pip install -r requirements.txt
Write-Host "`n[SUCCESS] Python venv setup complete!" -ForegroundColor Green
Write-Host "To test: python main_federated_flower.py --config configs/default.toml --subsample 60 --no_wandb"
