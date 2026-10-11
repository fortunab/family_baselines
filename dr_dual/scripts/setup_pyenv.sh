#!/usr/bin/env bash
# Setup Pyenv Environment on Linux / Ubuntu / WSL2
set -e

echo "=========================================================="
echo " Setting up Pyenv for Dual-Database DR & ISIC/HAM10000 ML "
echo "=========================================================="

PYTHON_VERSION="3.11.9"

echo "[1/3] Ensuring Python $PYTHON_VERSION is installed via pyenv..."
pyenv install -s "$PYTHON_VERSION"

echo "[2/3] Setting local Python version..."
pyenv local "$PYTHON_VERSION"

echo "[3/3] Creating and activating virtual environment..."
python -m venv "./.venv"
source "./.venv/bin/activate"
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "=========================================================="
echo " Pyenv setup complete! Activate with: source ./.venv/bin/activate"
echo "=========================================================="
