#!/usr/bin/env bash
# Setup script using Pyenv for Linux / WSL / macOS
set -e

TARGET_VERSION=$(cat .python-version | tr -d '[:space:]')

if ! command -v pyenv &> /dev/null; then
    echo "Error: pyenv is not installed. Install from: https://github.com/pyenv/pyenv" >&2
    exit 1
fi

echo "Setting local Python to $TARGET_VERSION..."
pyenv local "$TARGET_VERSION"

python -m venv venv_cervical_skorch
source venv_cervical_skorch/bin/activate
pip install -r requirements.txt

echo "[SUCCESS] pyenv virtual environment configured!"
