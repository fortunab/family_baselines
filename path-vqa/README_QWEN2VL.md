# Foundation Model 3: Alibaba Qwen2-VL

Alibaba's Vision-Language foundation model (`Qwen/Qwen2-VL-2B-Instruct`) adapted for high-resolution pathology visual reasoning on **Hugging Face `flaviagiammarino/path-vqa`**.

---

## 🔬 Architecture Highlights
- Native dynamic resolution visual encoder capable of resolving fine microscopic cellular features.
- Multi-turn clinical dialogue and reasoning capabilities.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\path_vqa_foundation_baseline
python main_path_vqa.py --model-name qwen2vl
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\path_vqa_foundation_baseline
source venv/bin/activate
python3 main_path_vqa.py --model-name qwen2vl
```
