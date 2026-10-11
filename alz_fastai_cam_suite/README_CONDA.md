# Conda / Mamba Environment Guide: Alzheimer MRI CAM Suite

Instructions for managing dependencies with **Anaconda**, **Miniconda**, or **Mamba**.

---

## 1. Environment Specification (`environment.yml`)
The environment configuration includes Python 3.11, PyTorch, fastai, and all clinical metrics and visualization packages:

```yaml
name: alzheimer-cam
channels:
  - fastai
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.11.9
  - pip
  - pip:
    - torch>=2.2.0
    - torchvision>=0.17.0
    - fastai>=2.7.14
    - scikit-learn>=1.4.0
    - scipy>=1.12.0
    - pillow>=10.2.0
    - matplotlib>=3.8.0
    - pandas>=2.2.0
    - pyyaml>=6.0.1
    - ruff>=0.4.0
    - flake8>=7.0.0
    - black>=24.3.0
```

---

## 2. Automated Script Setup

### Windows (PowerShell)
```powershell
.\scripts\setup_conda.ps1
```

### Linux / WSL (Bash)
```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
```

---

## 3. Manual Creation & Activation
```bash
# 1. Create the environment
conda env create -f environment.yml

# 2. Activate
conda activate alzheimer-cam

# 3. Verify packages
python -c "import fastai, torch; print('fastai:', fastai.__version__, '| PyTorch:', torch.__version__)"
```

---

## 4. Execution Commands
```bash
# Run one-line heatmap visualizer and metrics evaluation
python visualize_cam_heatmaps.py

# Run model training & Step 1 metrics
python train_and_evaluate.py

# Run quality checks
python run_linter.py
```
