# Herlev Cervical Cytology Benchmark Suite (70/15/15 Split)

This repository provides reproducible implementations of the 4 benchmark paradigms adapted for the **Herlev Cervical Cytology Dataset** from Kaggle:
👉 **[Kaggle: yuvrajsinhachowdhury/herlev-dataset](https://www.kaggle.com/datasets/yuvrajsinhachowdhury/herlev-dataset)**

---

## 📥 How to Download the Kaggle Dataset

### Option A: Automatic via Kaggle CLI
```bash
# 1. Download directly into ./data folder
python download_kaggle_herlev.py

# Or via direct Kaggle command:
kaggle datasets download -d yuvrajsinhachowdhury/herlev-dataset -p ./data --unzip
```

### Option B: Manual Browser Download
1. Download the archive from [Kaggle](https://www.kaggle.com/datasets/yuvrajsinhachowdhury/herlev-dataset).
2. Unzip into `herlev_cervical_baseline/data/`.
3. The dataset loader in [`herlev_dataset.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/herlev_dataset.py) automatically:
   - Recognizes all 7 class folders (with fuzzy alias matching).
   - **Filters out ground-truth mask files** (`-d.bmp`, `-cyt.bmp`, `_mask.png`) so only pure cell images are ingested.
   - Applies the **70% Train / 15% Val / 15% Test** split with dynamic random seeds.

---

## 🔬 The 7 Herlev Cytology Classes

| Index | Class Folder (Kaggle Aliases) | Cytological Diagnosis | Severity Category |
| :---: | :--- | :--- | :---: |
| **0** | `01_normal_superficial` (`superficial`, `normal_superficiel`) | Normal Superficial Squamous | **Normal** |
| **1** | `02_normal_intermediate` (`intermediate`, `normal_intermediate`) | Normal Intermediate Squamous | **Normal** |
| **2** | `03_normal_columnar` (`columnar`, `normal_columnar`) | Normal Columnar Endocervical | **Normal** |
| **3** | `04_mild_dysplastic` (`light_dysplastic`, `mild_dysplastic`) | Mild Dysplasia / CIN 1 / LSIL | **Abnormal** |
| **4** | `05_moderate_dysplastic` (`moderate_dysplastic`) | Moderate Dysplasia / CIN 2 / HSIL | **Abnormal** |
| **5** | `06_severe_dysplastic` (`severe_dysplastic`) | Severe Dysplasia / CIN 3 / HSIL | **Abnormal** |
| **6** | `07_carcinoma_in_situ` (`carcinoma_in_situ`, `carcinoma`) | Carcinoma in Situ / Malignant | **Abnormal** |

---

## 🏆 4-Tier Benchmark Leaderboard

| Tier | Paradigm | Architecture / Model | Evaluation Protocol | Detailed Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Classical ML Baseline** | **Handcrafted (LBP+GLCM+Gabor+Color) + RBF-SVM** | 70/15/15 Holdout Split | [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/README_SVM.md) |
| **2** | **Modern CNN Baseline** | **ConvNeXt (ConvNeXt-Tiny / ConvNeXt-Small)** | 70/15/15 Holdout Split | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/README_CONVNEXT.md) |
| **3** | **SOTA Vision Transformer** | **Vision Transformer (EVA-02 / ViT-Base / Swin)** | 70/15/15 Holdout Split | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/README_VIT.md) |
| **4** | **Pathology & Vision Foundation** | **Owkin Phikon / Paige Virchow / DINOv2** | 70/15/15 Holdout Split | [README_FOUNDATION.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline/README_FOUNDATION.md) |

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
# Navigate to the project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\herlev_cervical_baseline

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
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/herlev_cervical_baseline

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
