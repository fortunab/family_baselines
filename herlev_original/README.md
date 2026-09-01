# Original MDE-Lab Herlev Pap Smear Cytology Benchmark Suite (70/15/15 Split)

This repository provides reproducible implementations of the 4 benchmark paradigms adapted specifically for the **original Herlev Pap Smear Database from MDE-Lab**:
👉 **[MDE-Lab Downloads: https://mde-lab.aegean.gr/index.php/downloads/](https://mde-lab.aegean.gr/index.php/downloads/)**

---

## 📥 How to Download the MDE-Lab Dataset

### Option A: Automatic via Python Script
```bash
python download_mde_herlev.py
```

### Option B: Manual Download from MDE-Lab
1. Visit [MDE-Lab Downloads](https://mde-lab.aegean.gr/index.php/downloads/).
2. Under **Databases**, download the **Herlev Pap Smear Database** zip file.
3. Extract all folders into `herlev_original_mde_baseline/data/`.
4. The dataset loader in [`mde_dataset.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline/mde_dataset.py) automatically:
   - Recognizes all 7 original MDE-Lab category folders.
   - **Filters out ground-truth nucleus/cytoplasm mask files** (`*-d.bmp`, `*-cyt.bmp`) so only clean cell images are fed to the models.
   - Applies the **70% Train / 15% Val / 15% Test** stratified split with dynamic random seeds.

---

## 🔬 The 7 Original MDE-Lab Cytology Categories (917 Images)

| Class Index | MDE-Lab Folder Name | Cytological Diagnosis | Normal / Abnormal |
| :---: | :--- | :--- | :---: |
| **0** | `normal_superficiel` (74 cells) | Normal Superficial Squamous | **Normal** |
| **1** | `normal_intermediate` (70 cells) | Normal Intermediate Squamous | **Normal** |
| **2** | `columnar` (98 cells) | Normal Columnar Epithelial | **Normal** |
| **3** | `light_dysplastic` (182 cells) | Light / Mild Dysplasia (LSIL / CIN 1) | **Abnormal** |
| **4** | `moderate_dysplastic` (146 cells) | Moderate Dysplasia (HSIL / CIN 2) | **Abnormal** |
| **5** | `severe_dysplastic` (197 cells) | Severe Dysplasia (HSIL / CIN 3) | **Abnormal** |
| **6** | `carcinoma_in_situ` (150 cells) | Carcinoma in Situ / Malignant | **Abnormal** |

---

## 🏆 4-Tier Benchmark Leaderboard

| Tier | Paradigm | Architecture / Model | Evaluation Protocol | Detailed Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Classical ML Baseline** | **Handcrafted (LBP+GLCM+Gabor+Color) + RBF-SVM** | 70/15/15 Holdout Split | [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline/README_SVM.md) |
| **2** | **Modern CNN Baseline** | **ConvNeXt (ConvNeXt-Tiny / ConvNeXt-Small)** | 70/15/15 Holdout Split | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline/README_CONVNEXT.md) |
| **3** | **SOTA Vision Transformer** | **Vision Transformer (EVA-02 / ViT-Base / Swin)** | 70/15/15 Holdout Split | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline/README_VIT.md) |
| **4** | **Pathology & Vision Foundation** | **Owkin Phikon / Paige Virchow / DINOv2** | 70/15/15 Holdout Split | [README_FOUNDATION.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline/README_FOUNDATION.md) |

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
# Navigate to the project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_original_mde_baseline

# Tier 1: Classical SVM (70/15/15 Split)
python main_svm.py

# Tier 2: Modern ConvNeXt-Tiny CNN (70/15/15 Split)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer / EVA-02 (70/15/15 Split)
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Pathology Foundation Model (Owkin Phikon / Virchow)
python main_foundation.py --model-name owkin/phikon

# 4-Tier Comparison Leaderboard
python compare_baselines.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
# Navigate to the project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/herlev_original_mde_baseline

# Activate virtual environment
source venv/bin/activate

# Tier 1: Classical SVM
python3 main_svm.py

# Tier 2: ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Foundation Model + Linear Probe
python3 main_foundation.py --model-name owkin/phikon

# Compare all 4 baselines side-by-side
python3 compare_baselines.py
```
