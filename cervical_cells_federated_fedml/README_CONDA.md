# Conda Environment Guide: FedML Cervical Cytology

This guide covers setting up and executing the FedML Cervical Cytology suite using Anaconda, Miniconda, or Mamba.

---

## 1. Environment Creation

Create the dedicated Conda environment from `environment.yml`:
```powershell
conda env create -f environment.yml
```

Or execute the provided bootstrap script:
- **Windows**:
  ```powershell
  .\scripts\setup_conda.ps1
  ```
- **Linux / WSL**:
  ```bash
  bash scripts/setup_conda.sh
  ```

---

## 2. Activation

Activate the environment:
```powershell
conda activate cervical_cells_fedml
```

---

## 3. Verifying GPU Availability

Ensure CUDA is visible to PyTorch:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Running Experiments

```powershell
# Default fastai ConvNeXt run
python main_federated_fedml.py

# skorch ResNet run with FedProx
python main_federated_fedml.py --config configs/skorch_phikon.toml --strategy FedProx
```
