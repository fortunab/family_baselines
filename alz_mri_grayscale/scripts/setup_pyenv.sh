#!/usr/bin/env bash
# Setup script for pyenv on Linux / Ubuntu / WSL
set -euo pipefail

echo "==> Setting up Python 3.11.9 with pyenv..."
pyenv install 3.11.9 -s
pyenv local 3.11.9

echo "==> Creating virtual environment..."
python -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

echo "==> pyenv setup completed successfully!"
