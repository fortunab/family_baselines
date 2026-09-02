# Running PolypGen2.0 Hugging Face Detection on Ubuntu / WSL (COCO Protocol)

This guide walks you through setting up and running all 4 Foundation Object Detectors on the **Hugging Face `halyusuf/PolypGen2.0` dataset** inside **Ubuntu (WSL/WSL2)**.

---

## 1. Access the Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/polypgen_hf_detection_baseline
```

---

## 2. Environment Setup

```bash
# 1. Update system packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running Foundation Object Detectors

```bash
# Model 1: Microsoft Florence-2
python3 main_detection.py --model-name florence2

# Model 2: Google OWLv2
python3 main_detection.py --model-name owlv2

# Model 3: Grounding DINO
python3 main_detection.py --model-name grounding_dino

# Model 4: Google DeepMind PaliGemma
python3 main_detection.py --model-name paligemma

# Or run all 4 sequentially:
python3 main_detection.py --model-name all

# Generate 4-Model Comparison Leaderboard & Chart:
python3 compare_detectors.py
```
