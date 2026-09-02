# Foundation Model 2: NVIDIA SegFormer-B3

NVIDIA's Hierarchical Vision Transformer semantic segmentation model (`nvidia/segformer-b3-finetuned-ade-512-512`) adapted for binary polyp lesion segmentation on **Hugging Face `kowndinya23/Kvasir-SEG`**.

---

## 🔬 Architecture Highlights
- Multi-scale Mix Transformer (MiT) encoder extracting hierarchical multi-frequency features without positional embeddings.
- Lightweight All-MLP Decoder for real-time inference on endoscopy streams.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_seg_foundation_baseline
python main_segmentation.py --model-name segformer --epochs 10 --batch-size 4
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_seg_foundation_baseline
source venv/bin/activate
python3 main_segmentation.py --model-name segformer --epochs 10 --batch-size 4
```
