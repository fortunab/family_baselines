#!/usr/bin/env bash
# ==============================================================================
# Pyenv Automated Setup Script for Linux / Ubuntu / WSL
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

set -e

echo "======================================================================"
echo "   SUBSTRA CERVICAL FL: PYENV ENVIRONMENT SETUP"
echo "======================================================================"

if command -v pyenv &> /dev/null; then
    echo "[Pyenv-Setup] Pinned Python version (.python-version): 3.11.9"
    pyenv install 3.11.9 --skip-existing
    pyenv local 3.11.9
else
    echo "Warning: pyenv not found in PATH. Proceeding with system python3..."
fi

python3 -m venv .venv_pyenv
./.venv_pyenv/bin/pip install --upgrade pip
./.venv_pyenv/bin/pip install -r requirements.txt

echo ""
echo "======================================================================"
echo "   PYENV SETUP COMPLETE!"
echo "======================================================================"
echo "To activate and run:"
echo "  source .venv_pyenv/bin/activate"
echo "  python main_federated_substra.py --config configs/default.toml --framework fastai"
echo "  python main_federated_substra.py --config configs/default.toml --framework skorch"
