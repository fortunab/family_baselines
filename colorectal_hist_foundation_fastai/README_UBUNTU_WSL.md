# Ubuntu & WSL2 Setup Guide: fastai Pathology Foundation Models

This guide walks you through setting up an isolated Linux virtual environment, verifying the linter, and running experiments with **fastai Pathology Foundation Models** and **Weights & Biases** inside **Ubuntu (WSL/WSL2)**.

---

## 1. Access the Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_pathology_foundation_fastai
```

---

## 2. Virtual Environment Setup

```bash
# 1. Update packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 2. Create isolated virtual environment
python3 -m venv venv_foundation_fastai

# 3. Activate virtual environment
source venv_foundation_fastai/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Code Quality & Linter Verification

```bash
python3 run_linter.py
```

---

## 4. Running fastai Foundation Models with TOML Configs & W&B

```bash
# Model 1: Owkin Phikon
python3 main_fastai_foundation.py --config configs/phikon.toml

# Model 2: Paige Virchow
python3 main_fastai_foundation.py --config configs/virchow.toml

# Model 3: Harvard UNI
python3 main_fastai_foundation.py --config configs/uni.toml

# Model 4: Meta DINOv2
python3 main_fastai_foundation.py --config configs/dinov2.toml

# Comparison Leaderboard & Chart
python3 compare_foundation_models.py
```
