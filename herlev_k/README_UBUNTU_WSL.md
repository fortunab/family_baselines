# Running Herlev Cervical Cytology Baselines on Ubuntu / WSL (70/15/15 Split)

This guide walks you through setting up and running all 4 benchmark models on the **Herlev Cervical Cytology dataset** inside **Ubuntu (WSL/WSL2)** using the **70% Train / 15% Val / 15% Test** evaluation protocol.

---

## 1. Access the Herlev Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline
```

---

## 2. Setting Up Python Environment

```bash
# 1. Update packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Models

```bash
# Tier 1: Classical SVM (70/15/15 Split)
python3 main_svm.py

# Tier 2: ConvNeXt-Tiny (70/15/15 Split)
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer (70/15/15 Split)
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Foundation Model + Linear Probe (70/15/15 Split)
python3 main_foundation.py --model-name owkin/phikon

# 4-Tier Leaderboard Comparison
python3 compare_baselines.py
```

---

## 4. Running in Background (`nohup` / `tmux`)

```bash
# Run training in background
nohup python3 main_convnext.py --model-name convnext_tiny > convnext_herlev.log 2>&1 &

# Monitor logs live
tail -f convnext_herlev.log
```
