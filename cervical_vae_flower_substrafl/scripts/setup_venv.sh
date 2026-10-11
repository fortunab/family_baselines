#!/usr/bin/env bash
# Standard venv Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Python venv for Flower & Substra VAE Cytology  "
echo "=========================================================="

echo "[*] Creating standard virtual environment at .venv..."
python3 -m venv .venv

echo "[*] Upgrading pip and installing requirements..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo -e "\n[SUCCESS] Standard venv environment configured!"
echo "To activate:        source .venv/bin/activate"
echo "To execute Flower:  python main_federated_flower.py --preprocessing vae"
echo "To execute Substra: python main_federated_substra.py --preprocessing vae"
