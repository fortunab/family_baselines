#!/usr/bin/env bash
# ==============================================================================
# Conda / Mamba Automated Setup Script for Linux / Ubuntu / WSL
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

set -e

echo "======================================================================"
echo "   SUBSTRA CERVICAL FL: CONDA ENVIRONMENT SETUP"
echo "======================================================================"

if ! command -v conda &> /dev/null; then
    echo "Conda not found in PATH. Please install Miniconda or activate conda."
    exit 1
fi

echo "[Conda-Setup] Creating conda environment 'cervical_fl_substra'..."
conda env create -f environment.yml --overwrite

echo ""
echo "======================================================================"
echo "   CONDA SETUP COMPLETE!"
echo "======================================================================"
echo "To activate and run:"
echo "  conda activate cervical_fl_substra"
echo "  python main_federated_substra.py --config configs/default.toml --framework fastai"
echo "  python main_federated_substra.py --config configs/default.toml --framework skorch"
echo "  python compare_substra_strategies.py --rounds 3"
echo "  python run_linter.py"
