#!/usr/bin/env bash
# Pyenv Environment Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Pyenv for FedML Cytology                       "
echo "=========================================================="

TARGET_VERSION=$(cat .python-version | tr -d '[:space:]')
echo "[*] Pyenv target version: $TARGET_VERSION"

if command -v pyenv &> /dev/null; then
    echo "[*] Installing Python $TARGET_VERSION via pyenv..."
    pyenv install --skip-existing "$TARGET_VERSION"
    pyenv local "$TARGET_VERSION"
else
    echo "[!] pyenv not found in system path. Please ensure Python $TARGET_VERSION is installed."
fi

echo "[*] Creating virtual environment with active Python..."
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo -e "\n[SUCCESS] Pyenv environment setup complete!"
echo "To activate: source .venv/bin/activate"
