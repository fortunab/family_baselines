#!/usr/bin/env bash
# Pyenv Environment Setup Script for Linux / macOS / WSL
set -e

echo "=========================================================="
echo " Setting up Pyenv Environment for FedML Cervical Cytology  "
echo "=========================================================="

TARGET_PYTHON="3.11.9"

if ! command -v pyenv &> /dev/null; then
    echo "[!] pyenv not found. Please install pyenv: https://github.com/pyenv/pyenv"
    exit 1
fi

echo "[*] Installing Python $TARGET_PYTHON via pyenv..."
pyenv install -s $TARGET_PYTHON

echo "[*] Setting local directory Python version to $TARGET_PYTHON..."
pyenv local $TARGET_PYTHON

echo "[*] Initializing dedicated virtualenv..."
python -m venv venv_pyenv_fedml
source venv_pyenv_fedml/bin/activate
pip install -r requirements.txt

echo -e "\n[SUCCESS] Pyenv environment setup complete!"
echo "Run fastai:    python main_federated_fedml.py --framework fastai"
echo "Run skorch:    python main_federated_fedml.py --framework skorch"
