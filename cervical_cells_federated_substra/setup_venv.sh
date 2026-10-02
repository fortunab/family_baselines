#!/usr/bin/env bash
# ==============================================================================
# Standard Python venv Automated Setup Script for Linux / Ubuntu / WSL
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

set -e

echo "======================================================================"
echo "   SUBSTRA CERVICAL FL: PYTHON VENV SETUP"
echo "======================================================================"

echo "[venv-Setup] Creating virtual environment in .venv..."
python3 -m venv .venv

echo "[venv-Setup] Upgrading pip and wheel inside .venv..."
./.venv/bin/pip install --upgrade pip wheel setuptools

echo "[venv-Setup] Installing dependencies from requirements.txt..."
./.venv/bin/pip install -r requirements.txt

echo ""
echo "======================================================================"
echo "   VENV SETUP COMPLETE!"
echo "======================================================================"
echo "To activate and run:"
echo "  source .venv/bin/activate"
echo "  python main_federated_substra.py --config configs/default.toml --framework fastai"
echo "  python main_federated_substra.py --config configs/default.toml --framework skorch"
echo "  python compare_substra_strategies.py --rounds 3"
echo "  python run_linter.py"
