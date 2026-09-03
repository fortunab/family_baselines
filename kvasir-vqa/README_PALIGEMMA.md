# Foundation Model 2: Google DeepMind PaliGemma

Google DeepMind's sequence-to-sequence multimodal foundation model (`google/paligemma-3b-pt-224` / `448`) adapted for visual question answering on **Hugging Face `SimulaMet-HOST/Kvasir-VQA`**.

---

## 🔬 Architecture Highlights
- SigLIP-So400M Vision Transformer image encoder fused with Gemma-2B autoregressive language model.
- Prompt format: `answer en {question}` producing natural diagnostic answers on GI endoscopy.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_vqa_foundation_baseline
python main_kvasir_vqa.py --model-name paligemma
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_vqa_foundation_baseline
source venv/bin/activate
python3 main_kvasir_vqa.py --model-name paligemma
```
