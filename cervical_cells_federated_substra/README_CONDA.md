# Conda Environment Guide: Substra Cervical Cytology

Full environment specification managed via **Conda** or **Mamba**.

---

## 1. Automated Setup

### Windows PowerShell:
```powershell
.\setup_conda.ps1
```

### Linux / Ubuntu / WSL:
```bash
chmod +x setup_conda.sh
./setup_conda.sh
```

---

## 2. Activation and Execution

```powershell
conda activate cervical_fl_substra

# Run fastai
python main_federated_substra.py --config configs/default.toml --framework fastai

# Run skorch
python main_federated_substra.py --config configs/default.toml --framework skorch

# Benchmark
python compare_substra_strategies.py --rounds 3

# Linting
python run_linter.py
```
