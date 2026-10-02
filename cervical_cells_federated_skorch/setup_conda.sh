#!/usr/bin/env bash
# Setup script using Conda / Mamba for Linux / macOS
set -e

ENV_NAME="cervical_cells_skorch"

if command -v mamba &> /dev/null; then
    echo "Creating environment using Mamba..."
    mamba env create -f environment.yml
elif command -v conda &> /dev/null; then
    echo "Creating environment using Conda..."
    conda env create -f environment.yml
else
    echo "Error: Neither Conda nor Mamba was found in PATH." >&2
    exit 1
fi

echo "[SUCCESS] Conda environment '$ENV_NAME' created!"
echo "Activate with: conda activate $ENV_NAME"
