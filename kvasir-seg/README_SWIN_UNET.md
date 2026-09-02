# Foundation Model 4: Microsoft Swin-Transformer U-Net

Microsoft's Shifted Window Transformer foundation model (`microsoft/swin-base-patch4-window7-224`) structured in a symmetrical encoder-decoder U-Net for binary polyp lesion segmentation on **Hugging Face `kowndinya23/Kvasir-SEG`**.

---

## 🔬 Architecture Highlights
- Shifted window self-attention providing long-range context with linear computational complexity.
- Superior global shape modeling of large pedunculated colorectal polyps.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_seg_foundation_baseline
python main_segmentation.py --model-name swin_unet --epochs 10 --batch-size 4
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_seg_foundation_baseline
source venv/bin/activate
python3 main_segmentation.py --model-name swin_unet --epochs 10 --batch-size 4
```
