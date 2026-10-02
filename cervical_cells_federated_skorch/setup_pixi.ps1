# Setup script using Pixi for Windows (Self-Bootstrapping)
$ErrorActionPreference = "Stop"

Write-Host "=================================================================" -ForegroundColor Cyan
Write-Host "   PIXI ENVIRONMENT SETUP & DEPENDENCY RESOLUTION (SKORCH)" -ForegroundColor Cyan
Write-Host "=================================================================" -ForegroundColor Cyan

# 1. Check if pixi is already in PATH
if (-not (Get-Command pixi -ErrorAction SilentlyContinue)) {
    $pixiBin = "$HOME\.pixi\bin"
    if (Test-Path "$pixiBin\pixi.exe") {
        Write-Host "Found Pixi at $pixiBin. Adding to current session PATH..." -ForegroundColor Yellow
        $env:Path = "$pixiBin;$env:Path"
    } else {
        Write-Host "Pixi not found. Downloading and installing official Pixi binary..." -ForegroundColor Yellow
        Invoke-WebRequest -useb https://pixi.sh/install.ps1 | Invoke-Expression
        $env:Path = "$pixiBin;$env:Path"
    }
}

# 2. Verify pixi is operational
Write-Host "`nPixi version detected: $(pixi --version)" -ForegroundColor Green

# 3. Install environment and dependencies from pixi.toml
Write-Host "`nInstalling environment and dependencies with Pixi..." -ForegroundColor Cyan
pixi install

Write-Host "`n=================================================================" -ForegroundColor Green
Write-Host " [SUCCESS] Pixi environment ready for Flower + skorch!" -ForegroundColor Green
Write-Host "=================================================================" -ForegroundColor Green
Write-Host "To run fast dry-run: pixi run dry-run" -ForegroundColor Yellow
Write-Host "To run training:     pixi run train" -ForegroundColor Yellow
Write-Host "To run benchmark:    pixi run benchmark" -ForegroundColor Yellow
Write-Host "To run linting:      pixi run lint" -ForegroundColor Yellow
Write-Host "To enter shell:      pixi shell" -ForegroundColor Yellow
