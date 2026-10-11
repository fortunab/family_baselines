#!/usr/bin/env bash
# Setup script for Python venv on Linux / Ubuntu / WSL
set -euo pipefail

echo "==> Creating virtual environment in ./venv..."
python3 -m venv venv

echo "==> Activating virtual environment..."
source venv/bin/activate

echo "==> Upgrading pip and installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "==> Setup complete! Environment is ready."
