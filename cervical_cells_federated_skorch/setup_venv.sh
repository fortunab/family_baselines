#!/usr/bin/env bash
# Setup script using Standard Python venv for Linux / WSL / macOS
set -e

VENV_NAME="venv_cervical_skorch"

if [ ! -d "$VENV_NAME" ]; then
    echo "Creating virtual environment in $VENV_NAME..."
    python3 -m venv "$VENV_NAME"
fi

source "$VENV_NAME/bin/activate"
echo "Upgrading pip and installing requirements.txt..."
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

echo "[SUCCESS] Virtual environment ready in $VENV_NAME"
echo "To run training: python main_federated_skorch.py --config configs/default.toml"
