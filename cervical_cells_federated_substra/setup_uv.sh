#!/usr/bin/env bash
# ==============================================================================
# uv Automated Setup Script for Linux / Ubuntu / WSL
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

set -e

echo "======================================================================"
echo "   SUBSTRA CERVICAL FL: UV ENVIRONMENT SETUP"
echo "======================================================================"

if ! command -v uv &> /dev/null; then
    echo "[uv-Bootstrap] Installing standalone uv binary..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$PATH"
fi

echo "[uv-Setup] Creating virtual environment (.venv) using Python 3.11..."
uv venv .venv --python 3.11

echo "[uv-Setup] Installing dependencies via uv pip..."
uv pip install -e .

echo ""
echo "======================================================================"
echo "   UV SETUP COMPLETE!"
echo "======================================================================"
echo "To execute tasks:"
echo "  uv run python main_federated_substra.py --config configs/default.toml --framework fastai"
echo "  uv run python main_federated_substra.py --config configs/default.toml --framework skorch"
echo "  uv run python compare_substra_strategies.py --rounds 3"
echo "  uv run python run_linter.py"
