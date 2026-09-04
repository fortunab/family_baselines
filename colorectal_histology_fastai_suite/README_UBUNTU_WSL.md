# Running Colorectal Histology fastai & skorch Benchmark on Ubuntu / WSL2

This guide provides step-by-step instructions for creating a virtual environment and executing the benchmark inside **Ubuntu (WSL / WSL2 or Linux Server)**.

---

## 1. Access Project Directory in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_fastai_suite
```

---

## 2. Linux System Packages & Virtual Environment Setup

```bash
# 1. Update apt & install Python development tools
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0

# 2. Create isolated virtual environment
python3 -m venv venv_histology_fastai

# 3. Activate virtual environment
source venv_histology_fastai/bin/activate

# 4. Upgrade pip and install requirements
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running Benchmark Experiments

```bash
# fastai ConvNeXt-Base
python3 main_fastai.py --config configs/fastai_convnext.ini

# fastai Vision Transformer
python3 main_fastai.py --config configs/fastai_vit.ini

# skorch ResNet50
python3 main_skorch.py --config configs/skorch_resnet.ini

# Generate Leaderboard
python3 compare_suite.py
```
