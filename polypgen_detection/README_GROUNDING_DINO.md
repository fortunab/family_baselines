# Foundation Model 3: Grounding DINO (Cross-Modal Detection Transformer)

IDEA-Research's cross-modal detection transformer combining DINO DETR with language feature enhancement, evaluated on **Hugging Face `halyusuf/PolypGen2.0`**.

---

## 🔬 Architecture Highlights
- Dual-encoder architecture fusing multi-scale image features and BERT text tokens with cross-attention.
- SOTA bounding box localization on complex colon mucosa folds.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_hf_detection_baseline
python main_detection.py --model-name grounding_dino
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\polypgen_hf_detection_baseline
source venv/bin/activate
python3 main_detection.py --model-name grounding_dino
```
