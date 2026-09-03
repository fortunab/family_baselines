# Foundation Model 1: Microsoft Florence-2

Unified prompt-based vision-language foundation model (`microsoft/Florence-2-large` / `base`) adapted for visual question answering on **Hugging Face `SimulaMet-HOST/Kvasir-VQA`**.

---

## 🔬 Architecture Highlights
- Multi-task sequence-to-sequence foundation architecture prompted with `<VQA> {question}`.
- Advanced spatial grounding for colonoscopy lesion location and mucosal landmarks.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_vqa_foundation_baseline
python main_kvasir_vqa.py --model-name florence2
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_vqa_foundation_baseline
source venv/bin/activate
python3 main_kvasir_vqa.py --model-name florence2
```
