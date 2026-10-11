#!/usr/bin/env bash
# Pixi Environment Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Pixi for Flower & Substra VAE Cytology         "
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
echo "To run Flower with VAE:   pixi run test-flower-vae"
echo "To run Substra with VAE:  pixi run test-substra-vae"
echo "To run comparison:        pixi run compare"
