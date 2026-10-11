# Alzheimer MRI fastai CAM Suite: Linux / Ubuntu & WSL2 Guide

Complete execution and environment setup instructions for **Ubuntu 20.04/22.04 LTS** and **WSL2** (Windows Subsystem for Linux).

---

## 1. System Requirements & System Packages
Ensure Python 3.11, development headers, and build essentials are installed:
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential libgl1-mesa-glx
```

If utilizing NVIDIA GPUs on WSL2 or native Linux, verify CUDA driver availability:
```bash
nvidia-smi
```

---

## 2. Environment Setup

### Option A: Standard `venv` (Shell Script)
```bash
# Make setup script executable and run
chmod +x scripts/setup_venv.sh
./scripts/setup_venv.sh

# Or manual execution:
python3.11 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Option B: Pyenv
```bash
# Install and set local Python version
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option C: Conda / Mamba
```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh

# Or manual execution:
conda env create -f environment.yml
conda activate alzheimer-cam
```

---

## 3. Running Workflows on Linux / WSL

### 1-Line Execution (CAM Heatmaps & Multi-Step Metrics)
```bash
python visualize_cam_heatmaps.py
```

### Step 1: Model Training & 10 Clinical Classification Metrics
```bash
python train_and_evaluate.py --epochs 3 --lr 0.001 --device cpu
```
*(Specify `--device cuda` if an NVIDIA GPU is available).*

### Step 2: Multi-Method CAM Quantitative Benchmark
```bash
python compare_cam_metrics.py --samples-per-class 5
```

---

## 4. Code Quality & Linters
Run the 4-stage validation pipeline on Linux:
```bash
python run_linter.py
```
Outputs validation statuses for AST, Ruff, Flake8, and Black formatting.

---

## 5. Artifact Inspection
Generated images and JSON metric reports are placed in:
- `results/heatmaps/*.png`
- `results/metrics/*.json`

To copy or view outputs from WSL2 directly in Windows File Explorer:
```bash
explorer.exe results/heatmaps
```
