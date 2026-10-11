#!/usr/bin/env bash
# Setup script for Conda on Linux / Ubuntu / WSL
set -euo pipefail

echo "==> Creating Conda environment 'alzheimer-grayscale-bench'..."
conda env create -f environment.yml

echo "==> Conda environment created! Activate with: conda activate alzheimer-grayscale-bench"
