# Setup script using Pyenv for Windows
$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   PYENV ENVIRONMENT SETUP (SKORCH)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

$targetVersion = Get-Content ".python-version" -Raw
$targetVersion = $targetVersion.Trim()

if (-not (Get-Command pyenv -ErrorAction SilentlyContinue)) {
    Write-Error "pyenv-win is not found in PATH. Install from: https://github.com/pyenv-win/pyenv-win"
}

Write-Host "Setting local python version to $targetVersion..." -ForegroundColor Cyan
pyenv local $targetVersion

$pythonPath = (pyenv which python)
Write-Host "Active Python under pyenv: $pythonPath" -ForegroundColor Green

python -m venv venv_cervical_skorch
& ".\venv_cervical_skorch\Scripts\Activate.ps1"
pip install -r requirements.txt

Write-Host "[SUCCESS] pyenv environment ready!" -ForegroundColor Green
