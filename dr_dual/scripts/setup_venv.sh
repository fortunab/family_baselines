#!/usr/bin/env bash
# Setup Python Virtual Environment on Linux / Ubuntu / WSL2
set -e

echo "=========================================================="
echo " Setting up venv for Dual-Database DR & ISIC/HAM10000 ML "
echo "=========================================================="

VENV_PATH="./venv"

if [ ! -d "$VENV_PATH" ]; then
    echo "[1/4] Creating virtual environment at $VENV_PATH..."
    python3 -m venv "$VENV_PATH"
else
    echo "[1/4] Virtual environment already exists at $VENV_PATH."
fi

echo "[2/4] Activating virtual environment..."
source "$VENV_PATH/bin/activate"

echo "[3/4] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "[4/4] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo "=========================================================="
echo " Setup Complete! Activate with: source ./venv/bin/activate"
echo "=========================================================="
