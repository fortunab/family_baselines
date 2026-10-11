#!/usr/bin/env bash
# Pixi Environment Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Pixi for FedML Cytology (N2N & CycleGAN)      "
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
echo "To execute FedML with N2N:       pixi run python main_federated_fedml.py --preprocessing n2n"
echo "To execute FedML with CycleGAN:  pixi run python main_federated_fedml.py --preprocessing cyclegan"
echo "To execute FedML Dual Pipeline:  pixi run python main_federated_fedml.py --preprocessing n2n_cyclegan"
echo "To run Preprocessing Benchmark:  pixi run python compare_cervical_preprocessing.py"
