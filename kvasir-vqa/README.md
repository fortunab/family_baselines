# Gastrointestinal Endoscopy Visual Question Answering (Kvasir-VQA) Foundation Benchmark

This repository provides reproducible implementations of **4 performant Multimodal Foundation Models** evaluated on the official **[`SimulaMet-HOST/Kvasir-VQA`](https://huggingface.co/datasets/SimulaMet-HOST/Kvasir-VQA)** dataset on Hugging Face (6,500+ gastrointestinal and colorectal endoscopy frame question-answer pairs).

---

## 🔬 Dataset Schema & Endoscopic Question Typology

The dataset contains clinical visual question answering pairs over colonoscopy and GI endoscopy frames:
- `image`: RGB colonoscopy endoscopy frame $[H, W, 3]$.
- `question`: Clinical query (*"Is there a polyp in this endoscopy image?"*, *"Where is the polyp located?"*, *"What instrument is visible?"*, *"Which anatomical landmark is shown?"*).
- `answer`: Ground-truth diagnosis (closed-ended `"yes"`/`"no"` or open-ended phrases like `"center"`, `"snare"`, `"cecum"`, `"z-line"`).
- `source`: Source cohort (`HyperKvasir` or `Kvasir-Instrument`).

---

## 🏆 The 4 Foundation VQA Models

| Model # | Foundation Architecture | Hugging Face ID / Base | Dedicated Guide |
| :---: | :--- | :--- | :--- |
| **1** | **Microsoft Florence-2** | `microsoft/Florence-2-large` / `base` (Prompt-Guided `<VQA>`) | [README_FLORENCE2.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_vqa_foundation_baseline/README_FLORENCE2.md) |
| **2** | **Google DeepMind PaliGemma** | `google/paligemma-3b-pt-224` / `448` (SigLIP + Gemma-2B) | [README_PALIGEMMA.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_vqa_foundation_baseline/README_PALIGEMMA.md) |
| **3** | **Alibaba Qwen2-VL** | `Qwen/Qwen2-VL-2B-Instruct` (Dynamic High-Res ViT + Qwen2) | [README_QWEN2VL.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_vqa_foundation_baseline/README_QWEN2VL.md) |
| **4** | **Salesforce BLIP-2** | `Salesforce/blip2-opt-2.7b` (Q-Former Multimodal Bridge) | [README_BLIP2.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/kvasir_vqa_foundation_baseline/README_BLIP2.md) |

---

## 📊 Clinical VQA Evaluation Metrics
- **Overall VQA Accuracy (Exact Match / EM %)**: String equality across all GI questions.
- **Closed-Ended (Yes/No) Diagnostic Metrics**: Accuracy, Precision, Recall, F1-Score on binary diagnostic queries.
- **Open-Ended Linguistic Metrics**:
  - **BLEU-1 & BLEU-4**: N-gram precision for endoscopy anatomical and pathology terminology.
  - **ROUGE-L F1**: Longest common subsequence matching.
  - **Semantic Token F1**: Token-level precision and recall.
- **Diagnostic Visual QA Card**: Endoscopic image + Question + Ground Truth + Model Prediction.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\kvasir_vqa_foundation_baseline

# 1. Model 1: Microsoft Florence-2
python main_kvasir_vqa.py --model-name florence2

# 2. Model 2: Google DeepMind PaliGemma
python main_kvasir_vqa.py --model-name paligemma

# 3. Model 3: Alibaba Qwen2-VL
python main_kvasir_vqa.py --model-name qwen2vl

# 4. Model 4: Salesforce BLIP-2
python main_kvasir_vqa.py --model-name blip2

# Or run all 4 sequentially:
python main_kvasir_vqa.py --model-name all

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_kvasir_vqa.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\kvasir_vqa_foundation_baseline
source venv/bin/activate

python3 main_kvasir_vqa.py --model-name florence2
python3 main_kvasir_vqa.py --model-name paligemma
python3 main_kvasir_vqa.py --model-name qwen2vl
python3 main_kvasir_vqa.py --model-name blip2
python3 compare_kvasir_vqa.py
```
