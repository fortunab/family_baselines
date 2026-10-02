#!/usr/bin/env bash
# Setup script using standard Python venv for Linux / WSL
set -e
echo "Creating Python venv environment 'venv_flower_cervical'..."
python3 -m venv venv_flower_cervical
source venv_flower_cervical/bin/activate
pip install --upgrade pip setuptools wheel
echo "Installing requirements..."
pip install -r requirements.txt
echo "[SUCCESS] Python venv setup complete!"
