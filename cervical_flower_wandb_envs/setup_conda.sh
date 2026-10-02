#!/usr/bin/env bash
# Setup script using Conda / Mamba for Linux / WSL
set -e
echo "Creating Conda environment 'cervical_flower_env' from environment.yml..."
conda env create -f environment.yml
echo "[SUCCESS] Conda environment created!"
echo "To activate: conda activate cervical_flower_env"
