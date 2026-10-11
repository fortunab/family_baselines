#!/usr/bin/env bash
# uv Environment Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up uv Virtual Environment for FedML Cytology     "
echo "=========================================================="

if ! command -v uv &> /dev/null; then
    echo "[*] uv not found. Installing via official installer..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[*] Creating virtual environment with Python 3.11..."
uv venv --python 3.11 .venv

echo "[*] Synchronizing dependencies using uv pip..."
uv pip install -r requirements.txt

echo -e "\n[SUCCESS] uv environment configured at .venv!"
echo "To activate:        source .venv/bin/activate"
echo "To execute FedML:   uv run python main_federated_fedml.py --preprocessing n2n_cyclegan"
