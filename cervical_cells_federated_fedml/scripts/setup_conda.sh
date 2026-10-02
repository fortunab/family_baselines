#!/usr/bin/env bash
# Conda Environment Setup Script for Linux / macOS / WSL
set -e

echo "=========================================================="
echo " Setting up Conda Environment for FedML Cervical Cytology  "
echo "=========================================================="

ENV_NAME="cervical_cells_fedml"

echo "[*] Creating Conda environment from environment.yml..."
conda env create -f environment.yml --overwrite

echo -e "\n[SUCCESS] Conda environment '$ENV_NAME' created successfully!"
echo "To activate:   conda activate $ENV_NAME"
echo "Run fastai:    python main_federated_fedml.py --framework fastai"
echo "Run skorch:    python main_federated_fedml.py --framework skorch"
