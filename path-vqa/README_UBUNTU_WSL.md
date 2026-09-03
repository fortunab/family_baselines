# Running PathVQA Foundation Benchmark on Ubuntu / WSL (Linux)

This guide walks you through setting up and running all 4 Multimodal Foundation Models on the **Hugging Face `flaviagiammarino/path-vqa` dataset** inside **Ubuntu (WSL/WSL2)**.

---

## 1. Access the Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/path_vqa_foundation_baseline
```

---

## 2. Environment Setup

```bash
# 1. Update packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 2. Virtual Environment
python3 -m venv venv
source venv/bin/activate

# 3. Dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running VQA Foundation Models

```bash
# Model 1: Microsoft BiomedCLIP / Florence-2
python3 main_path_vqa.py --model-name biomedclip

# Model 2: Google DeepMind PaliGemma
python3 main_path_vqa.py --model-name paligemma

# Model 3: Alibaba Qwen2-VL
python3 main_path_vqa.py --model-name qwen2vl

# Model 4: Salesforce BLIP-2
python3 main_path_vqa.py --model-name blip2

# Comparison Leaderboard & Chart
python3 compare_vqa_models.py
```
