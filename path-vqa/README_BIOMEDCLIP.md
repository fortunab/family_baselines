# Foundation Model 1: Microsoft BiomedCLIP / Florence-2

Biomedical vision-language foundation model pre-trained on 15 million PubMed Central image-text pairs, adapted for visual question answering on **Hugging Face `flaviagiammarino/path-vqa`**.

---

## 🔬 Architecture Highlights
- Heavy ViT vision encoder + PubMedBERT language representation.
- Fine-grained semantic alignment with histology and pathology terms.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\path_vqa_foundation_baseline
python main_path_vqa.py --model-name biomedclip
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline
source venv/bin/activate
python3 main_path_vqa.py --model-name biomedclip
```
