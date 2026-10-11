# Setup script for pyenv-win on Windows (PowerShell)
$ErrorActionPreference = "Stop"

Write-Host "Configuring Python 3.11.9 with pyenv-win..." -ForegroundColor Cyan
pyenv install 3.11.9 -SkipExisting
pyenv local 3.11.9

Write-Host "Creating virtual environment..." -ForegroundColor Cyan
python -m venv venv
& ".\venv\Scripts\Activate.ps1"

python -m pip install --upgrade pip
pip install -r requirements.txt

Write-Host "pyenv-win setup completed successfully!" -ForegroundColor Green
