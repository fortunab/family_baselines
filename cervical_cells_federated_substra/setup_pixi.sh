#!/usr/bin/env bash
# ==============================================================================
# Pixi Automated Setup Script for Linux / Ubuntu / WSL
# Cervical Cytology Federated Substra Suite (fastai + skorch)
# ==============================================================================

set -e

echo "======================================================================"
echo "   SUBSTRA CERVICAL FL: PIXI ENVIRONMENT SETUP"
echo "======================================================================"

if ! command -v pixi &> /dev/null; then
    if [ -f "$HOME/.pixi/bin/pixi" ]; then
        export PATH="$HOME/.pixi/bin:$PATH"
        echo "[Pixi-Bootstrap] Found Pixi at $HOME/.pixi/bin. Added to PATH."
    else
        echo "[Pixi-Bootstrap] Pixi not detected. Installing official Pixi binary..."
        curl -fsSL https://pixi.sh/install.sh | bash
        export PATH="$HOME/.pixi/bin:$PATH"
    fi
fi

echo "[Pixi-Setup] Solving and installing dependencies via pixi.toml..."
pixi install

echo ""
echo "======================================================================"
echo "   PIXI SETUP COMPLETE!"
echo "======================================================================"
echo "To execute tasks:"
echo "  pixi run train-fastai   # Run Substra training with fastai"
echo "  pixi run train-skorch   # Run Substra training with skorch"
echo "  pixi run dry-run        # Fast dry-run with Random Best Select"
echo "  pixi run benchmark      # Compare strategies & frameworks"
echo "  pixi run lint           # Static code quality checker"
