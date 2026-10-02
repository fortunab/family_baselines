# Windows Native Deployment & Execution Guide

This document covers native Windows execution details, PyTorch CUDA compatibility, OpenMP threading considerations, and path handling for the Cervical Cytology Flower Suite.

---

## 1. Native Windows Considerations

### 1.1 Intel OpenMP Multiple Runtimes Error Fix (`libiomp5md.dll`)
When PyTorch, NumPy, and OpenCV are loaded simultaneously on Windows, Intel OpenMP can throw:
`OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.`

**Built-in Fix**:
`src/__init__.py` automatically injects:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
No manual intervention is required.

### 1.2 Ray & Flower Multiprocessing on Windows
Flower uses Ray or multiprocessing under Windows. On Windows, spawn is used rather than fork.
Ensure all execution entry points are protected by:
```python
if __name__ == "__main__":
    main()
```
Both `main_federated_flower.py` and `compare_federated_strategies.py` strictly adhere to this standard.

### 1.3 Windows Path Lengths (MAX_PATH)
Windows natively supports 260 characters in paths unless Long Paths are enabled in the registry.
The code uses `pathlib.Path` with normalized forward slashes and relative project subdirectories to prevent path overflow issues.

---

## 2. Windows PowerShell Commands

### Activate existing environment:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\venv_cervical_flower\Scripts\Activate.ps1
```

### Run quick verification:
```powershell
python main_federated_flower.py --config configs/default.toml --num_rounds 2 --num_clients 3 --local_epochs 1 --subsample 60 --no_wandb
```

### Check GPU status:
```powershell
nvidia-smi
```
