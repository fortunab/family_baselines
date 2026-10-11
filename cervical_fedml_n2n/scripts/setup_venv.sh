#!/usr/bin/env bash
# Standard venv Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Python venv for FedML Cytology                 "
echo "=========================================================="

echo "[*] Creating standard virtual environment at .venv..."
python3 -m venv .venv

echo "[*] Upgrading pip and installing requirements..."
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt

echo -e "\n[SUCCESS] Standard venv environment configured!"
echo "To activate:        source .venv/bin/activate"
echo "To execute FedML:   python main_federated_fedml.py --preprocessing n2n_cyclegan"
