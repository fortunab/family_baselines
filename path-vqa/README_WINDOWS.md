# Running PathVQA Foundation Benchmark on Windows (PowerShell & CMD)

This guide walks you through setting up and running all 4 Multimodal Foundation Models on the **Hugging Face `flaviagiammarino/path-vqa` dataset** on **Windows 10/11**.

---

## 1. Environment & Setup

Open **PowerShell** or **Command Prompt** and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\path_vqa_foundation_baseline

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running Individual Foundation VQA Models

### Model 1: Microsoft BiomedCLIP / Florence-2
```powershell
python main_path_vqa.py --model-name biomedclip
```

### Model 2: Google DeepMind PaliGemma
```powershell
python main_path_vqa.py --model-name paligemma
```

### Model 3: Alibaba Qwen2-VL
```powershell
python main_path_vqa.py --model-name qwen2vl
```

### Model 4: Salesforce BLIP-2
```powershell
python main_path_vqa.py --model-name blip2
```

---

## 3. Running All Models & Leaderboard Generation

```powershell
# Sequentially evaluate all 4 models:
python main_path_vqa.py --model-name all

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_vqa_models.py
```

All metrics summaries (`.json`) and diagnostic visual QA cards (`.png`) are saved in the `results/` folder.
