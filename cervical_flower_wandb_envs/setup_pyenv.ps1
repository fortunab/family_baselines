# Setup script using pyenv-win on Windows
$PyVersion = Get-Content .python-version -Raw
$PyVersion = $PyVersion.Trim()
Write-Host "Installing/setting Python version $PyVersion via pyenv..." -ForegroundColor Cyan
pyenv install $PyVersion -s
pyenv local $PyVersion
Write-Host "Creating dedicated virtual environment..." -ForegroundColor Cyan
python -m venv venv_flower_cervical
.\venv_flower_cervical\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
Write-Host "`n[SUCCESS] Pyenv environment setup complete!" -ForegroundColor Green
