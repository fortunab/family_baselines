#!/usr/bin/env bash
set -euo pipefail

# Run from the directory containing this script.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Create and activate the virtual environment.
if [[ ! -d venv ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

# Install dependencies.
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Run code-quality checks.
python run_linter.py

# Train all foundation models sequentially.
python main_herlev_skorch.py --config configs/phikon.toml
python main_herlev_skorch.py --config configs/virchow.toml
python main_herlev_skorch.py --config configs/dinov2.toml
python main_herlev_skorch.py --config configs/biomedclip.toml

echo "All training runs completed successfully."
