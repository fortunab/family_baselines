# Alzheimer MRI Grayscale Benchmark: Linux / Ubuntu & WSL2 Guide

Complete environment setup and execution instructions for **Ubuntu 20.04/22.04 LTS** and **WSL2**.

---

## 1. System Requirements & Packages
Install Python 3.11 development packages:
```bash
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev build-essential libgl1-mesa-glx
```

Verify GPU availability (optional for CPU execution):
```bash
nvidia-smi
```

---

## 2. Environment Setup

### Option A: Standard `venv`
```bash
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
pyenv install 3.11.9
pyenv local 3.11.9
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Option C: Conda
```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
conda activate alzheimer-grayscale-bench
```

---

## 3. Running Workflows on Linux

### One-Line Grand Benchmark Execution
```bash
python run_full_benchmark_suite.py
```

### Individual Steps
```bash
# 1. Grayscale Preprocessing
python preprocess_grayscale_dataset.py

# 2. Classical ML Family Experiments
python run_ml_family_experiment.py

# 3. Foundational Vision Backbone Comparison
python run_foundational_comparison.py

# 4. Code Quality Linter
python run_linter.py
```
