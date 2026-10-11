# Ubuntu & WSL2 Linux Execution Guide

This guide details configuring and running Suite 27 (`cervical_cells_vae_flower_substra`) on Linux environments (Ubuntu 22.04 / 24.04 LTS) and Windows Subsystem for Linux (WSL2).

---

## 🐧 Prerequisites & Setup

```bash
# Update base utilities
sudo apt-get update && sudo apt-get install -y git build-essential libgl1-mesa-glx libglib2.0-0

# Clone or navigate to the suite folder
cd cervical_cells_vae_flower_substra
```

### Choose an Environment Manager (pixi, conda, or venv only)
```bash
# Option A: Pixi (Recommended)
curl -fsSL https://pixi.sh/install.sh | bash
pixi install

# Option B: Conda
bash ./scripts/setup_conda.sh
conda activate cervical_cells_vae_fs

# Option C: Python venv
bash ./scripts/setup_venv.sh
source venv_cervical_vae/bin/activate
```

---

## 🏃 Headless Plotting & Execution

On headless servers or WSL2 without an active display server, Matplotlib runs cleanly in non-interactive `Agg` mode.

```bash
# Run Flower Federated Learning
python main_federated_flower.py --framework fastai --preprocessing vae --rounds 5

# Run Substra Federated Learning
python main_federated_substra.py --framework skorch --preprocessing vae --rounds 5

# Run Preprocessing Comparison
python compare_cervical_vae.py

# Run Quality Gate
python run_linter.py
```
