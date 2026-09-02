# Running Kvasir-SEG Foundation Segmentation on Windows (PowerShell & CMD)

This guide walks you through setting up and running all 4 Foundation Segmentation Models on the **Hugging Face `kowndinya23/Kvasir-SEG` dataset** on **Windows 10/11**.

---

## 1. Environment & Setup

Open **PowerShell** or **Command Prompt** and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_seg_foundation_baseline

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running Individual Foundation Segmentation Models

### Model 1: MedSAM / SAM
```powershell
python main_segmentation.py --model-name medsam --epochs 10 --batch-size 4
```

### Model 2: NVIDIA SegFormer-B3
```powershell
python main_segmentation.py --model-name segformer --epochs 10 --batch-size 4
```

### Model 3: Meta ConvNeXt-UNet
```powershell
python main_segmentation.py --model-name convnext_unet --epochs 10 --batch-size 4
```

### Model 4: Microsoft Swin-UNet
```powershell
python main_segmentation.py --model-name swin_unet --epochs 10 --batch-size 4
```

---

## 3. Running All Models & Leaderboard Generation

```powershell
# Sequentially run all 4 models:
python main_segmentation.py --model-name all --epochs 10

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_seg_models.py
```

All checkpoints (`.pt`), summary metrics (`.json`), and 4-panel segmentation grids (`.png`) are saved in the `results/` folder.
