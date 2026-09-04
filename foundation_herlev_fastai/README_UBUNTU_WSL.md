# Ubuntu & Linux / WSL2 Setup Guide: Herlev Cytology Foundation Suite

Step-by-step instructions for running the **Herlev Cervical Cytology Pathology Foundation fastai Suite** on Linux environments (Ubuntu 20.04/22.04 LTS, Debian, RedHat, or WSL2 on Windows 11).

---

## 1. System Packages

```bash
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv libgl1-mesa-glx libglib2.0-0 git
```

---

## 2. Virtual Environment Setup

```bash
cd herlev_pathology_foundation_fastai

# Create dedicated virtualenv
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip setuptools wheel

# Install PyTorch with CUDA (or CPU)
# CUDA 12.1:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install suite requirements
pip install -r requirements.txt
```

---

## 3. Verify Code Quality

```bash
python run_linter.py
```

---

## 4. Run Model Training

```bash
# Owkin Phikon (iBOT ViT-Base)
python main_herlev_fastai.py --config configs/phikon.toml

# Paige Virchow (ViT-Huge 632M)
python main_herlev_fastai.py --config configs/virchow.toml

# Harvard UNI (ViT-Large)
python main_herlev_fastai.py --config configs/uni.toml

# Meta DINOv2
python main_herlev_fastai.py --config configs/dinov2.toml

# Microsoft BiomedCLIP
python main_herlev_fastai.py --config configs/biomedclip.toml
```

---

## 5. Weights & Biases (W&B) Telemetry

```bash
export WANDB_API_KEY="your_api_key"
python main_herlev_fastai.py --config configs/phikon.toml
```

For offline execution:
```bash
python main_herlev_fastai.py --config configs/phikon.toml --no_wandb
```

---

## 6. Generate Leaderboard

```bash
python compare_herlev_models.py
```
Output artifacts are saved in `results/`.
