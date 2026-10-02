#!/usr/bin/env bash
# Setup script using pyenv on Linux / macOS
set -e
PY_VER=$(cat .python-version | tr -d '[:space:]')
echo "Installing/setting Python version $PY_VER via pyenv..."
pyenv install -s $PY_VER
pyenv local $PY_VER
echo "Creating dedicated virtual environment..."
python -m venv venv_flower_cervical
source venv_flower_cervical/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
echo "[SUCCESS] Pyenv environment setup complete!"
