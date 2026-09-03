# Ubuntu & WSL2 Setup Guide: fastai + TOML + W&B Suite

This guide walks you through setting up an isolated Linux virtual environment, verifying the linter, and running experiments with **Weights & Biases** inside **Ubuntu (WSL/WSL2)**.

---

## 1. Access the Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_toml_wandb
```

---

## 2. Virtual Environment Setup

```bash
# 1. Update packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 2. Create isolated virtual environment
python3 -m venv venv_fastai_toml

# 3. Activate virtual environment
source venv_fastai_toml/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running fastai Models with TOML Configs & W&B

```bash
# Model 1: ConvNeXt-Base
python3 main_fastai_wandb.py --config configs/convnext_base.toml

# Model 2: Vision Transformer (ViT-Base)
python3 main_fastai_wandb.py --config configs/vit_base.toml

# Model 3: EfficientNetV2
python3 main_fastai_wandb.py --config configs/efficientnet_v2.toml

# Comparison Leaderboard & Chart
python3 compare_fastai_models.py
```
