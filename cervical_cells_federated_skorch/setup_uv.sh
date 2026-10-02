#!/usr/bin/env bash
# Setup script using Astral uv for Linux / WSL / macOS
set -e

if ! command -v uv &> /dev/null; then
    echo "uv not found. Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
fi

echo "uv version: $(uv --version)"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment via uv..."
    uv venv --python 3.11 .venv
fi

echo "Installing dependencies using uv pip..."
uv pip install -r requirements.txt

echo "[SUCCESS] uv environment ready in .venv"
echo "To run training: uv run python main_federated_skorch.py --config configs/default.toml"
