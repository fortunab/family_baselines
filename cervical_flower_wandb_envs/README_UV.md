# Astral UV Environment & Execution Guide

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package and environment manager written in Rust, serving as a 10-100x faster drop-in replacement for `pip`, `venv`, and `pip-tools`.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
.\setup_uv.ps1
```

### Linux / WSL2 / macOS Bash
```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

---

## 2. Using `uv` Directly

### Step 2.1: Install `uv`
```powershell
# Windows PowerShell:
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS:
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Step 2.2: Create and Sync Environment
```bash
# Create venv with specific Python:
uv venv --python 3.11 .venv

# Activate:
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
# Linux / macOS:
source .venv/bin/activate

# Install requirements with ultra-fast resolver:
uv pip install -r requirements.txt
```

---

## 3. Running with `uv run` (Zero-Activation Execution)

You can run scripts directly through `uv run` without activating the shell:

```bash
# Dry run verification
uv run python main_federated_flower.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb

# Full training run
uv run python main_federated_flower.py --config configs/convnext.toml

# Strategy benchmark
uv run python compare_federated_strategies.py --strategies FedAvg FedProx --num_rounds 5

# Linter suite
uv run python run_linter.py
```
