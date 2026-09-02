# Foundation Model 3: Meta ConvNeXt-Base U-Net

Meta's modern convolutional foundation model (`facebook/convnext-base-224-22k`) integrated into a Feature Pyramid U-Net decoder for binary polyp lesion segmentation on **Hugging Face `kowndinya23/Kvasir-SEG`**.

---

## 🔬 Architecture Highlights
- Inverted bottleneck blocks with large $7\times 7$ depthwise convolutions and Layer Normalization.
- Multi-scale skip connections preserving fine mucosal boundary contours.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_seg_foundation_baseline
python main_segmentation.py --model-name convnext_unet --epochs 10 --batch-size 4
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_seg_foundation_baseline
source venv/bin/activate
python3 main_segmentation.py --model-name convnext_unet --epochs 10 --batch-size 4
```
