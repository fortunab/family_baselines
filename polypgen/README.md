# PolypGen Colonoscopy Benchmark Suite (Synapse syn26376615)

This repository provides reproducible implementations of the 4 benchmark paradigms adapted for the **official Synapse PolypGen Benchmark (syn26376615)**:
👉 **[Synapse: syn26376615 / wiki/613312](https://www.synapse.org/Synapse:syn26376615/wiki/613312)**

---

## 📥 How to Download the Synapse Dataset

### Option A: Automatic via Python Script
```bash
# 1. Install synapseclient
pip install synapseclient

# 2. Run the automated downloader with your Synapse Personal Access Token (PAT)
python download_synapse_polypgen.py --auth-token YOUR_SYNAPSE_TOKEN
```
*(Get your free token at [Synapse.org Account Settings -> Personal Access Tokens](https://www.synapse.org/#!PersonalAccessTokens:))*.

### Option B: Manual Browser Download
1. Visit the [Synapse syn26376615 Files page](https://www.synapse.org/Synapse:syn26376615/files/).
2. Download the multi-center archives (`data_C1` .. `data_C6`).
3. Extract into `polypgen_baseline/data/`.
4. The dataset loader in [`polyp_dataset.py`](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline/polyp_dataset.py) automatically:
   - Discovers all center folders (`data_C1` to `data_C6`) and video sequence folders (`seq_*`).
   - Automatically maps `images/` to `1_POLYP` and `negative_only/` to `0_NO_POLYP`.
   - **Excludes all segmentation masks** (`masks/`, `*_mask.*`, `ground_truth/`).
   - Applies the **70% Train / 15% Val / 15% Test** stratified split with dynamic random seeds.

---

## 🔬 Synapse PolypGen Multi-Center Architecture

- **Total Frames**: 8,037 colonoscopy frames across 6 international medical centers:
  - `data_C1`: France (Positive & Negative frames)
  - `data_C2`: Italy (Positive & Negative frames)
  - `data_C3`: Norway (Positive & Negative frames)
  - `data_C4`: United Kingdom (Positive & Negative frames)
  - `data_C5`: Egypt (Positive & Negative frames)
  - `data_C6`: Multi-hospital sequence frames

---

## 🏆 4-Tier Benchmark Leaderboard

| Tier | Paradigm | Architecture / Model | Evaluation Protocol | Detailed Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Classical ML Baseline** | **Handcrafted (LBP+GLCM+Gabor+Color) + RBF-SVM** | 70/15/15 Holdout Split | [README_SVM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline/README_SVM.md) |
| **2** | **Modern CNN Baseline** | **ConvNeXt (ConvNeXt-Tiny / ConvNeXt-Small)** | 70/15/15 Holdout Split | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline/README_CONVNEXT.md) |
| **3** | **SOTA Vision Transformer** | **Vision Transformer (EVA-02 / ViT-Base / Swin)** | 70/15/15 Holdout Split | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline/README_VIT.md) |
| **4** | **Vision & Medical Foundation** | **DINOv2 / Phikon Foundation + Linear Probe** | 70/15/15 Holdout Split | [README_FOUNDATION.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline/README_FOUNDATION.md) |

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_baseline

# Tier 1: Classical SVM (70/15/15 Split)
python main_svm.py

# Tier 2: Modern ConvNeXt-Tiny CNN (70/15/15 Split)
python main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer / EVA-02 (70/15/15 Split)
python main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Foundation Model + Linear Probe (70/15/15 Split)
python main_foundation.py --model-name dinov2_base

# 4-Tier Comparison Leaderboard
python compare_baselines.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_baseline
source venv/bin/activate

python3 main_svm.py
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32
python3 main_foundation.py --model-name dinov2_base
python3 compare_baselines.py
```
