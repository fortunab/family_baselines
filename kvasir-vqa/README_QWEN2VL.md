# Foundation Model 3: Alibaba Qwen2-VL

Alibaba's Vision-Language foundation model (`Qwen/Qwen2-VL-2B-Instruct`) adapted for high-resolution endoscopy visual reasoning on **Hugging Face `SimulaMet-HOST/Kvasir-VQA`**.

---

## 🔬 Architecture Highlights
- Native dynamic resolution visual encoder capable of resolving fine mucosal surface pit patterns and instruments.
- Multi-turn clinical dialogue and reasoning capabilities.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_vqa_foundation_baseline
python main_kvasir_vqa.py --model-name qwen2vl
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_vqa_foundation_baseline
source venv/bin/activate
python3 main_kvasir_vqa.py --model-name qwen2vl
```
