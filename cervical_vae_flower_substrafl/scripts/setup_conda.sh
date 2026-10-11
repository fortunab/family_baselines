#!/usr/bin/env bash
# Conda Environment Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Conda for Flower & Substra VAE Cytology        "
echo "=========================================================="

ENV_NAME="cervical_cells_vae_flower_substra"

echo "[*] Creating Conda environment from environment.yml..."
conda env create -f environment.yml --name "$ENV_NAME" --yes

echo -e "\n[SUCCESS] Conda environment '$ENV_NAME' created!"
echo "To activate:        conda activate $ENV_NAME"
echo "To execute Flower:  python main_federated_flower.py --preprocessing vae"
echo "To execute Substra: python main_federated_substra.py --preprocessing vae"
