# CVC-Colon Colonoscopy Benchmark Suite (CVC-ClinicDB / CVC-ColonDB)

This repository provides reproducible implementations of the 4 benchmark paradigms adapted specifically for the **CVC-Colon Databases from the Computer Vision Center (UAB)**:
👉 **[CVC-Colon Databases: https://pages.cvc.uab.es/CVC-Colon/index.php/databases/](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/)**

---

## 📥 How to Download the CVC-Colon Dataset

### Option A: Automatic via Python Script
```bash
python download_cvc_colon.py
```

### Option B: Manual Download from CVC-Colon
1. Visit [CVC-Colon Databases](https://pages.cvc.uab.es/CVC-Colon/index.php/databases/).
2. Download **CVC-ClinicDB** (612 frames) or **CVC-ColonDB** (300 frames).
3. Extract into `cvc_colon_baseline/data/`.
4. The dataset loader in [`cvc_dataset.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cvc_colon_baseline/cvc_dataset.py) automatically:
   - Recursively indexes colonoscopy frames.
   - **Filters out ground-truth segmentation masks** (`Ground Truth/`, `masks/`, `*_mask.*`) so only camera RGB frames are ingested.
   - Applies the **70% Train / 15% Val / 15% Test** stratified split with dynamic random seeds.

---

## 🔬 Binary Classification Task

| Class Index | Class Name | Description |
| :---: | :--- | :--- |
| **0** | `0_NO_POLYP` | Normal Colon Mucosa / Non-Polyp Frame |
| **1** | `1_POLYP` | Polyp Present / Lesion Detected |

---

## 🏆 4-Tier Benchmark Leaderboard

| Tier | Paradigm | Architecture / Model | Evaluation Protocol | Detailed Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Classical ML Baseline** | **Handcrafted (LBP+GLCM+Gabor+Color) + RBF-SVM** | 70/15/15 Holdout Split | [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cvc_colon_baseline/README_SVM.md) |
| **2** | **Modern CNN Baseline** | **ConvNeXt (ConvNeXt-Tiny / ConvNeXt-Small)** | 70/15/15 Holdout Split | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cvc_colon_baseline/README_CONVNEXT.md) |
| **3** | **SOTA Vision Transformer** | **Vision Transformer (EVA-02 / ViT-Base / Swin)** | 70/15/15 Holdout Split | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cvc_colon_baseline/README_VIT.md) |
| **4** | **Vision & Medical Foundation** | **DINOv2 / Phikon + Linear Probe** | 70/15/15 Holdout Split | [README_FOUNDATION.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/cvc_colon_baseline/README_FOUNDATION.md) |

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
# Navigate to the project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\cvc_colon_baseline

# Tier 1: Classical SVM (70/15/15 Split)
python main_svm.py

# Tier 2: Modern ConvNeXt-Tiny CNN (70/15/15 Split)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer / EVA-02 (70/15/15 Split)
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Foundation Model (DINOv2 / Phikon)
python main_foundation.py --model-name dinov2_base

# 4-Tier Comparison Leaderboard
python compare_baselines.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
# Navigate to the project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/cvc_colon_baseline

# Activate virtual environment
source venv/bin/activate

# Tier 1: Classical SVM
python3 main_svm.py

# Tier 2: ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Foundation Model + Linear Probe
python3 main_foundation.py --model-name dinov2_base

# Compare all 4 baselines side-by-side
python3 compare_baselines.py
```
