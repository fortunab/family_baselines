#!/usr/bin/env bash
# Standard venv Environment Setup Script for Linux / macOS / WSL
set -e

echo "=========================================================="
echo " Setting up Python venv for FedML Cervical Cytology        "
echo "=========================================================="

VENV_DIR="venv_federated_fedml"

echo "[*] Creating standard virtual environment at '$VENV_DIR'..."
python3 -m venv $VENV_DIR

echo "[*] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

echo "[*] Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel

echo "[*] Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo -e "\n[SUCCESS] Virtual environment '$VENV_DIR' is ready!"
echo "To activate:   source $VENV_DIR/bin/activate"
echo "Run fastai:    python main_federated_fedml.py --framework fastai"
echo "Run skorch:    python main_federated_fedml.py --framework skorch"
