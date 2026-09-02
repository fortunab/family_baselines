# Foundation Model 1: Microsoft Florence-2 (`<OD>` Object Detection)

Microsoft's vision foundation model pre-trained on 5.4 billion region-text annotations, evaluated on **Hugging Face `halyusuf/PolypGen2.0`** using prompt `<OD>`.

---

## 🔬 Architecture Highlights
- Unified sequence-to-sequence prompt representation (`<OD>` $\rightarrow$ coordinates).
- DaViT multi-scale vision backbone paired with standard autoregressive text decoder.
- High precision on subtle flat mucosal lesions.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_hf_detection_baseline
python main_detection.py --model-name florence2
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline
source venv/bin/activate
python3 main_detection.py --model-name florence2
```
