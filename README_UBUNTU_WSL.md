# Running on Ubuntu / WSL (Windows Subsystem for Linux)

This guide walks you through setting up and running both the **Classical SVM Baseline** and the **SOTA Vision Transformer Baseline** inside **Ubuntu (WSL/WSL2)**.

---

## 1. Accessing the Code in WSL

You can either run the project directly from the Windows mount or copy it to the native Linux filesystem for faster I/O.

### Option A: Run directly from the Windows mount
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline
```

### Option B: Copy to native Linux home directory (Recommended for maximum speed)
```bash
cp -r /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline ~/colorectal_histology_svm_baseline
cd ~/colorectal_histology_svm_baseline
```

---

## 2. Setting Up Python Environment in Ubuntu

```bash
# 1. Update package list and install system tools
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies (PyTorch, timm, scikit-learn, etc.)
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Models

### 🔷 Baseline 1: Handcrafted Features + RBF-SVM (87.4%)
```bash
# Full 10-fold CV baseline
python3 main_svm.py

# With Grid Search hyperparameter optimization
python3 main_svm.py --tune-hyperparams
```

### 🔶 Baseline 2: SOTA Vision Transformer (98.4% - 99.17%)
```bash
# Quick dry run on 64 images (fast test)
python3 main_vit.py --subsample 64 --epochs 1 --batch-size 8

# Full ViT-Base fine-tuning
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# SOTA EVA-02 fine-tuning (448x448 resolution)
python3 main_vit.py --model-name eva02_base_patch14_448 --img-size 448 --epochs 15
```

### 📊 Baseline Comparison
```bash
# Compare results side-by-side
python3 compare_baselines.py
```

---

## 4. Running in Background (nohup / tmux)

```bash
# Start ViT training in background with nohup
nohup python3 main_vit.py --model-name vit_base_patch16_224 > vit_training.log 2>&1 &

# Monitor real-time training progress
tail -f vit_training.log
```

---

## 5. Viewing Output Results in Windows from WSL

Open the generated plots and metrics directly in Windows File Explorer:
```bash
explorer.exe results
```
