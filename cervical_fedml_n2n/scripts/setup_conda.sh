#!/usr/bin/env bash
# Conda Environment Setup Script for Linux / WSL / macOS
set -euo pipefail

echo "=========================================================="
echo " Setting up Conda Environment for FedML Cytology          "
echo "=========================================================="

ENV_NAME="cervical_cells_fedml_n2n_cyclegan"

echo "[*] Creating Conda environment from environment.yml..."
conda env create -f environment.yml --name "$ENV_NAME" --yes

echo -e "\n[SUCCESS] Conda environment '$ENV_NAME' created!"
echo "To activate:        conda activate $ENV_NAME"
echo "To execute FedML:   python main_federated_fedml.py --preprocessing n2n_cyclegan"
