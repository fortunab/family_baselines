# Foundation Model 4: Google DeepMind PaliGemma (Spatial Token Grounding)

Google DeepMind's open multimodal model pre-trained on dense spatial grounding tasks, evaluated on **Hugging Face `halyusuf/PolypGen2.0`**.

---

## 🔬 Architecture Highlights
- SigLIP vision backbone paired with Gemma 2B language decoder.
- Discretized spatial coordinate tokens (`<loc0000>` to `<loc1023>`) generating normalized boxes directly in text stream.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\polypgen_hf_detection_baseline
python main_detection.py --model-name paligemma
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\polypgen_hf_detection_baseline
source venv/bin/activate
python3 main_detection.py --model-name paligemma
```
