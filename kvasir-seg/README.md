# Kvasir-SEG Foundation Semantic Segmentation Benchmark (70/15/15 Split)

This repository provides reproducible implementations of **4 performant Foundation Semantic Segmentation Models** adapted for the official **[`kowndinya23/Kvasir-SEG`](https://huggingface.co/datasets/kowndinya23/Kvasir-SEG)** dataset on Hugging Face (1,000 dense image-mask pairs for gastrointestinal polyp segmentation).

---

## 🔬 Dataset Style: Dense Pixel-Wise Binary Segmentation

Unlike bounding-box detection datasets (`objects.bbox`), `kowndinya23/Kvasir-SEG` is a **dense pixel-level semantic segmentation dataset**:
- `image`: RGB Colonoscopy endoscopy frame $[H, W, 3]$.
- `annotation`: Full-resolution binary mask $[H, W]$ ($255 = \text{Polyp}$, $0 = \text{Mucosa background}$).

---

## 🏆 The 4 Foundation Segmentation Models

| Model # | Foundation Architecture | Backbone / Source | Dedicated Guide |
| :---: | :--- | :--- | :--- |
| **1** | **MedSAM / SAM** | `wanglab/medsam-vit-base` / `facebook/sam-vit-base` (Meta & MedSAM Foundation) | [README_MEDSAM.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_seg_foundation_baseline/README_MEDSAM.md) |
| **2** | **NVIDIA SegFormer-B3** | `nvidia/segformer-b3-finetuned-ade-512-512` / `nvidia/mit-b3` | [README_SEGFORMER.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_seg_foundation_baseline/README_SEGFORMER.md) |
| **3** | **Meta ConvNeXt-UNet** | `facebook/convnext-base-224-22k` + Feature Pyramid Decoder | [README_CONVNEXT_UNET.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_seg_foundation_baseline/README_CONVNEXT_UNET.md) |
| **4** | **Microsoft Swin-UNet** | `microsoft/swin-base-patch4-window7-224` (Shifted Window Attention) | [README_SWIN_UNET.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_seg_foundation_baseline/README_SWIN_UNET.md) |

---

## 📊 Clinical Segmentation Metrics & Loss Functions
- **Loss Function**: Combined Binary Cross-Entropy + Dice Loss ($\mathcal{L} = 0.5 \mathcal{L}_{\text{BCE}} + 0.5 \mathcal{L}_{\text{Dice}}$) for handling small/flat lesions.
- **Core Clinical Metrics**:
  - **Dice Similarity Coefficient (DSC / F1-Score)**: $\frac{2 |A \cap B|}{|A| + |B|}$
  - **Mean Intersection over Union (mIoU / Jaccard)**: $\frac{|A \cap B|}{|A \cup B|}$
  - **Pixel Accuracy (PA)**, Specificity, Precision, Recall (Lesion Sensitivity)
- **Visual Diagnostic Overlays**: 4-panel visual plots (Image, Ground Truth, Predicted Probability, Difference Map).
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_seg_foundation_baseline

# 1. Model 1: MedSAM / SAM
python main_segmentation.py --model-name medsam --epochs 10 --batch-size 4

# 2. Model 2: NVIDIA SegFormer-B3
python main_segmentation.py --model-name segformer --epochs 10 --batch-size 4

# 3. Model 3: Meta ConvNeXt-UNet
python main_segmentation.py --model-name convnext_unet --epochs 10 --batch-size 4

# 4. Model 4: Microsoft Swin-UNet
python main_segmentation.py --model-name swin_unet --epochs 10 --batch-size 4

# Or run all 4 sequentially:
python main_segmentation.py --model-name all --epochs 10

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_seg_models.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/kvasir_seg_foundation_baseline
source venv/bin/activate

python3 main_segmentation.py --model-name medsam --epochs 10 --batch-size 4
python3 main_segmentation.py --model-name segformer --epochs 10 --batch-size 4
python3 main_segmentation.py --model-name convnext_unet --epochs 10 --batch-size 4
python3 main_segmentation.py --model-name swin_unet --epochs 10 --batch-size 4
python3 compare_seg_models.py
```
