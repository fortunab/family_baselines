#!/usr/bin/env bash
# Setup script using Astral uv on Linux / macOS
set -e
echo "Creating virtual environment with uv..."
uv venv .venv --python 3.11
source .venv/bin/activate
echo "Installing dependencies with uv..."
uv pip install -r requirements.txt
echo "[SUCCESS] uv environment ready!"
echo "To run training: uv run python main_federated_flower.py --config configs/default.toml"
