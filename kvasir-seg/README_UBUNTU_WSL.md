# Running Kvasir-SEG Foundation Segmentation on Ubuntu / WSL (Linux)

This guide walks you through setting up and running all 4 Foundation Segmentation Models on the **Hugging Face `kowndinya23/Kvasir-SEG` dataset** inside **Ubuntu (WSL/WSL2)**.

---

## 1. Access the Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/kvasir_seg_foundation_baseline
```

---

## 2. Environment Setup

```bash
# 1. Update packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx

# 2. Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running Segmentation Models

```bash
# Model 1: MedSAM / SAM
python3 main_segmentation.py --model-name medsam --epochs 10 --batch-size 4

# Model 2: NVIDIA SegFormer-B3
python3 main_segmentation.py --model-name segformer --epochs 10 --batch-size 4

# Model 3: Meta ConvNeXt-UNet
python3 main_segmentation.py --model-name convnext_unet --epochs 10 --batch-size 4

# Model 4: Microsoft Swin-UNet
python3 main_segmentation.py --model-name swin_unet --epochs 10 --batch-size 4

# Comparison Leaderboard & Chart
python3 compare_seg_models.py
```
