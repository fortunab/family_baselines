# Ubuntu / Linux / WSL Quickstart Guide: Substra Cervical Cytology

Quickstart for Ubuntu, Debian, CentOS, and Windows Subsystem for Linux (WSL).

---

## 1. Environment Setup

```bash
# Option A: Pixi (Recommended)
chmod +x setup_pixi.sh
./setup_pixi.sh

# Option B: uv
chmod +x setup_uv.sh
./setup_uv.sh

# Option C: Conda
chmod +x setup_conda.sh
./setup_conda.sh

# Option D: venv
chmod +x setup_venv.sh
./setup_venv.sh
```

---

## 2. Launching Substra Simulation

```bash
# Run with fastai
python main_federated_substra.py --config configs/default.toml --framework fastai

# Run with skorch
python main_federated_substra.py --config configs/default.toml --framework skorch
```
