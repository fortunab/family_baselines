# Conda Environment Guide

This guide covers setup and execution within a native Anaconda or Miniconda environment.

---

## 🚀 Environment Creation

### Using Setup Script
```powershell
.\scripts\setup_conda.ps1
```

### Manual Creation
```powershell
conda env create -f environment.yml --name cervical_cells_fedml_n2n_cyclegan
conda activate cervical_cells_fedml_n2n_cyclegan
```

---

## 🏃 Running Experiments

```powershell
# FastAI with dual preprocessing
python main_federated_fedml.py --preprocessing n2n_cyclegan --framework fastai

# Skorch with CycleGAN harmonization
python main_federated_fedml.py --preprocessing cyclegan --framework skorch

# Preprocessing comparative benchmark
python compare_cervical_preprocessing.py
```
