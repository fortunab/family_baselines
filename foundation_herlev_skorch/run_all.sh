#!/usr/bin/env bash

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run code quality linter
python run_linter.py

# Train foundation models
python main_herlev_skorch.py --config configs/phikon.toml
python main_herlev_skorch.py --config configs/virchow.toml
python main_herlev_skorch.py --config configs/dinov2.toml
python main_herlev_skorch.py --config configs/biomedclip.toml


echo "All training runs completed successfully."
