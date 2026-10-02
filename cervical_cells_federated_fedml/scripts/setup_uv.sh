#!/usr/bin/env bash
# uv Environment Setup Script for Linux / macOS / WSL
set -e

echo "=========================================================="
echo " Setting up uv Environment for FedML Cervical Cytology     "
echo "=========================================================="

if ! command -v uv &> /dev/null; then
    echo "[*] uv not found. Installing via Astral official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[*] Creating virtual environment (.venv) using Python 3.11..."
uv venv --python 3.11 .venv

echo "[*] Installing project dependencies into .venv..."
source .venv/bin/activate
uv pip install -e ".[dev]"

echo -e "\n[SUCCESS] uv environment is ready!"
echo "Activate with: source .venv/bin/activate"
echo "Run fastai:    python main_federated_fedml.py --framework fastai"
echo "Run skorch:    python main_federated_fedml.py --framework skorch"
