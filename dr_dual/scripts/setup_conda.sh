#!/usr/bin/env bash
# Setup Conda Environment on Linux / Ubuntu / WSL2
set -e

echo "=========================================================="
echo " Setting up Conda for Dual-Database DR & ISIC/HAM10000 ML "
echo "=========================================================="

ENV_NAME="dr_dual_db_env"

echo "[1/2] Creating/Updating environment from environment.yml..."
if conda info --envs | grep -q "$ENV_NAME"; then
    echo "Environment exists, updating..."
    conda env update -f environment.yml -n "$ENV_NAME" --prune
else
    echo "Creating environment..."
    conda env create -f environment.yml -n "$ENV_NAME"
fi

echo "[2/2] Conda setup finished."
echo "=========================================================="
echo " Activate environment with: conda activate $ENV_NAME"
echo "=========================================================="
