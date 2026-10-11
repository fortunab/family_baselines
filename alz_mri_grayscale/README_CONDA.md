# Conda / Mamba Guide: Alzheimer MRI Grayscale Benchmark

Environment setup using **Anaconda**, **Miniconda**, or **Mamba**.

---

## 1. Environment Specification (`environment.yml`)
The environment configuration includes Python 3.11, PyTorch, fastai, and all ML/foundational packages:

```yaml
name: alzheimer-grayscale-bench
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
    - numpy>=1.26.0
    - pandas>=2.2.0
    - pillow>=10.2.0
    - matplotlib>=3.8.0
    - seaborn>=0.13.0
    - pyyaml>=6.0.1
    - timm>=0.9.16
    - ruff>=0.4.0
    - flake8>=7.0.0
    - black>=24.3.0
```

---

## 2. Automated Script Setup

### Windows
```powershell
.\scripts\setup_conda.ps1
conda activate alzheimer-grayscale-bench
```

### Linux / WSL
```bash
chmod +x scripts/setup_conda.sh
./scripts/setup_conda.sh
conda activate alzheimer-grayscale-bench
```

---

## 3. Execution
```bash
python run_full_benchmark_suite.py
```
