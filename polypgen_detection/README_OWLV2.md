# Foundation Model 2: Google OWLv2 (Open-Vocabulary ViT Detector)

Google's Open-Vocabulary Vision Transformer detector pre-trained on large-scale web image-text pairs, evaluated on **Hugging Face `halyusuf/PolypGen2.0`**.

---

## 🔬 Architecture Highlights
- Vision Transformer (ViT-Base / Patch16) backbone with classification head transferred to detection box coordinates.
- Open-vocabulary query embedding matching ("polyp", "colorectal polyp", "lesion").
- High zero-shot transferability to colonoscopy endoscopy frames.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_hf_detection_baseline
python main_detection.py --model-name owlv2
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\polypgen_hf_detection_baseline
source venv/bin/activate
python3 main_detection.py --model-name owlv2
```
