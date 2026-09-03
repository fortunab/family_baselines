# Foundation Model 4: Salesforce BLIP-2

Salesforce's multimodal foundation model (`Salesforce/blip2-opt-2.7b`) adapted for visual question answering on **Hugging Face `flaviagiammarino/path-vqa`**.

---

## 🔬 Architecture Highlights
- Two-stage pre-trained Querying Transformer (Q-Former) bridging vision encoder with LLMs.
- Zero-shot and few-shot clinical question-answering capabilities.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\path_vqa_foundation_baseline
python main_path_vqa.py --model-name blip2
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\path_vqa_foundation_baseline
source venv/bin/activate
python3 main_path_vqa.py --model-name blip2
```
