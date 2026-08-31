# Running on Ubuntu / WSL (Windows Subsystem for Linux)

This guide walks you through setting up and running all four benchmark paradigms inside **Ubuntu (WSL/WSL2)**:
1. **Classical SVM Baseline** (~87.4%)
2. **Modern ConvNeXt CNN Baseline** (~96.3% - 97.4%)
3. **SOTA Vision Transformer Baseline** (~98.4% - 99.17%)
4. **Computational Pathology Foundation (Virchow / Phikon)** (> 98.5% - 99.0%)

---

## 1. Accessing the Code in WSL

```bash
# Navigate to the project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline

# Or copy to Linux home for maximum I/O speed:
cp -r /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_svm_baseline ~/colorectal_histology_svm_baseline
cd ~/colorectal_histology_svm_baseline
```

---

## 2. Environment Setup

```bash
# 1. Update packages & install system tools
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

### 🔷 Tier 1: Classical SVM Baseline (87.4%)
```bash
python3 main_svm.py
```

### 🔶 Tier 2: Modern ConvNeXt CNN Baseline (96.3% - 97.4%)
```bash
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32
```

### 🟣 Tier 3: SOTA Vision Transformer Baseline (98.4% - 99.17%)
```bash
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32
```

### 🟢 Tier 4: Computational Pathology Foundation Model (> 98.5%)
```bash
# Paige Virchow
python3 main_virchow.py --model-name paige-ai/Virchow

# Owkin Phikon (Open-access)
python3 main_virchow.py --model-name owkin/phikon
```

### 📊 4-Way Comparative Leaderboard
```bash
python3 compare_baselines.py
```

---

## 4. Running in Background (nohup / tmux)

```bash
# Run Virchow in background
nohup python3 main_virchow.py --model-name owkin/phikon > virchow.log 2>&1 &

# Monitor logs
tail -f virchow.log
```

---

## 5. Viewing Output Results in Windows from WSL

```bash
explorer.exe results
```
