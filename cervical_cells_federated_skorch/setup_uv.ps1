# Setup script using Astral uv for Windows
$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   UV ENVIRONMENT SETUP & DEPENDENCY RESOLUTION (SKORCH)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check if uv is installed
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv not found. Downloading and installing uv..." -ForegroundColor Yellow
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:Path = "$HOME\.cargo\bin;$HOME\.local\bin;$env:Path"
}

Write-Host "`nuv version: $(uv --version)" -ForegroundColor Green

# 2. Create virtual environment using uv
if (-not (Test-Path ".venv")) {
    Write-Host "Creating virtual environment with Python 3.11 via uv..." -ForegroundColor Cyan
    uv venv --python 3.11 .venv
}

# 3. Synchronize / Install dependencies
Write-Host "Installing dependencies using uv pip..." -ForegroundColor Cyan
uv pip install -r requirements.txt

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] uv virtual environment ready in .venv" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "To run fast dry-run: uv run python main_federated_skorch.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb" -ForegroundColor Yellow
Write-Host "To run full training: uv run python main_federated_skorch.py --config configs/default.toml" -ForegroundColor Yellow
Write-Host "To run linting:      uv run python run_linter.py" -ForegroundColor Yellow
