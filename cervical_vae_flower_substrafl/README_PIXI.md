# Pixi Package & Environment Workflow Guide

Suite 27 provides native reproducible dependency management and task execution via **Pixi** (`pixi.toml`), supporting cross-platform execution on `win-64` and `linux-64`.

---

## 🚀 Quickstart with Pixi

### 1. Setup Environment & Lockfile
```powershell
# In cervical_cells_vae_flower_substra directory:
pixi install
```

This resolves conda-forge binary packages (Python 3.11, PyTorch, TorchVision, NumPy, SciPy, Scikit-Learn) and PyPI packages (Flower `flwr`, `skorch`, `fastai`, `wandb`) into a unified, reproducible `.pixi` environment.

---

## 🛠️ Built-in Pixi Tasks

Pre-configured tasks defined in `pixi.toml`:

```powershell
# Run 4-stage static linter
pixi run lint

# Run Flower Federated Learning with fastai
pixi run train-flower

# Run Substra Federated Learning with fastai
pixi run train-substra

# Run Flower with VAE preprocessing
pixi run train-vae-flower

# Run Substra with VAE preprocessing
pixi run train-vae-substra

# Benchmark Raw Cytology vs VAE-denoised representation
pixi run compare-vae
```

---

## 📋 Custom Execution in Pixi Shell

Activate the Pixi interactive shell or run commands directly:

```powershell
# Run arbitrary script inside Pixi environment
pixi run python main_federated_flower.py --framework skorch --preprocessing vae --rounds 3

pixi run python main_federated_substra.py --framework fastai --strategy FedProx --proximal_mu 0.5
```
