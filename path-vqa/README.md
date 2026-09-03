# Pathology Visual Question Answering (PathVQA) Foundation Benchmark

This repository provides reproducible implementations of **4 performant Multimodal Foundation Models** evaluated on the official **[`flaviagiammarino/path-vqa`](https://huggingface.co/datasets/flaviagiammarino/path-vqa)** dataset on Hugging Face (4,099 microscopic pathology image-question-answer triplets).

---

## 🔬 Dataset Schema & Diagnostic Question Typology

The dataset contains clinical visual question answering pairs over microscopic H&E pathology tiles:
- `image`: RGB pathology microscopy image $[H, W, 3]$.
- `question`: Clinical pathology inquiry (e.g. *"Is there malignant tissue present?"*, *"What type of cell is indicated?"*).
- `answer`: Ground-truth clinical diagnosis (both closed-ended `"yes"`/`"no"` and open-ended clinical entities).

---

## 🏆 The 4 Foundation VQA Models

| Model # | Foundation Architecture | Hugging Face ID / Base | Dedicated Guide |
| :---: | :--- | :--- | :--- |
| **1** | **Microsoft BiomedCLIP / Florence-2** | `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` | [README_BIOMEDCLIP.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline/README_BIOMEDCLIP.md) |
| **2** | **Google DeepMind PaliGemma** | `google/paligemma-3b-pt-224` / `448` (SigLIP + Gemma-2B) | [README_PALIGEMMA.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline/README_PALIGEMMA.md) |
| **3** | **Alibaba Qwen2-VL** | `Qwen/Qwen2-VL-2B-Instruct` (Dynamic High-Res ViT + Qwen2) | [README_QWEN2VL.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline/README_QWEN2VL.md) |
| **4** | **Salesforce BLIP-2** | `Salesforce/blip2-opt-2.7b` (Q-Former Multimodal Bridge) | [README_BLIP2.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline/README_BLIP2.md) |

---

## 📊 Clinical VQA Evaluation Metrics
- **Overall VQA Accuracy (Exact Match / EM %)**: String equality across all questions.
- **Closed-Ended (Yes/No) Diagnostic Metrics**: Accuracy, Precision, Recall, F1-Score on binary diagnostic queries.
- **Open-Ended Linguistic Metrics**:
  - **BLEU-1 & BLEU-4**: N-gram precision for clinical terminology.
  - **ROUGE-L F1**: Longest common subsequence matching.
  - **Semantic Token F1**: Token-level precision and recall.
- **Diagnostic Visual QA Card**: Microscopic tissue image + Question + Ground Truth + Model Prediction.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\path_vqa_foundation_baseline

# 1. Model 1: Microsoft BiomedCLIP / Florence-2
python main_path_vqa.py --model-name biomedclip

# 2. Model 2: Google DeepMind PaliGemma
python main_path_vqa.py --model-name paligemma

# 3. Model 3: Alibaba Qwen2-VL
python main_path_vqa.py --model-name qwen2vl

# 4. Model 4: Salesforce BLIP-2
python main_path_vqa.py --model-name blip2

# Or run all 4 sequentially:
python main_path_vqa.py --model-name all

# Generate 4-Model Comparison Leaderboard & Chart:
python compare_vqa_models.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline
source venv/bin/activate

python3 main_path_vqa.py --model-name biomedclip
python3 main_path_vqa.py --model-name paligemma
python3 main_path_vqa.py --model-name qwen2vl
python3 main_path_vqa.py --model-name blip2
python3 compare_vqa_models.py
```
