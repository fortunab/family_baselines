# Foundation Model 1: MedSAM / Segment Anything Model (SAM)

Meta and Medical SAM foundation model (`wanglab/medsam-vit-base`) adapted for binary polyp lesion segmentation on **Hugging Face `kowndinya23/Kvasir-SEG`**.

---

## 🔬 Architecture Highlights
- Heavy Vision Transformer (ViT-Base) image encoder pre-trained on 1.5 million+ medical image-mask pairs.
- High boundary fidelity on subtle sessile and flat colorectal adenomas.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_seg_foundation_baseline
python main_segmentation.py --model-name medsam --epochs 10 --batch-size 4
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_seg_foundation_baseline
source venv/bin/activate
python3 main_segmentation.py --model-name medsam --epochs 10 --batch-size 4
```
