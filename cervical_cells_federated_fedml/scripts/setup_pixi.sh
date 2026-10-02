#!/usr/bin/env bash
# Pixi Environment Setup Script for Linux / macOS / WSL
set -e

echo "=========================================================="
echo " Setting up Pixi Environment for FedML Cervical Cytology   "
echo "=========================================================="

if ! command -v pixi &> /dev/null; then
    echo "[*] Pixi not found. Installing via official installer..."
    curl -fsSL https://pixi.sh/install.sh | bash
    export PATH="$HOME/.pixi/bin:$PATH"
fi

echo "[*] Resolving and installing Pixi dependencies..."
pixi install

echo "[*] Verifying environment by running static quality checks..."
pixi run lint

echo -e "\n[SUCCESS] Pixi environment is ready!"
echo "To execute FedML with fastai:  pixi run python main_federated_fedml.py --framework fastai"
echo "To execute FedML with skorch:  pixi run python main_federated_fedml.py --framework skorch"
