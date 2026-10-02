# Setup script using Astral uv on Windows
Write-Host "Creating virtual environment with uv..." -ForegroundColor Cyan
uv venv .venv --python 3.11
Write-Host "Activating uv environment..." -ForegroundColor Cyan
.\.venv\Scripts\Activate.ps1
Write-Host "Installing dependencies with uv..." -ForegroundColor Cyan
uv pip install -r requirements.txt
Write-Host "`n[SUCCESS] uv environment ready!" -ForegroundColor Green
Write-Host "To run code check: uv run python run_linter.py" -ForegroundColor Yellow
Write-Host "To run training:   uv run python main_federated_flower.py --config configs/default.toml" -ForegroundColor Yellow
