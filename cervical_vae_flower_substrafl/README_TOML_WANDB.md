# TOML Configuration & Weights & Biases (W&B) MLOps Guide

Suite 27 integrates hierarchical TOML configuration files with enterprise MLOps tracking via **Weights & Biases (W&B)** and local JSON/CSV fallbacks.

---

## ⚙️ TOML Configuration Architecture

Configuration files reside in `configs/`:
- `configs/default_flower.toml`: Base configuration for Flower federated runs.
- `configs/default_substra.toml`: Base configuration for Substra federated runs.
- `configs/vae_flower_fastai.toml`: Flower + fastai + VAE preprocessing pipeline.
- `configs/vae_flower_skorch.toml`: Flower + skorch + VAE preprocessing pipeline.
- `configs/vae_substra_fastai.toml`: Substra + fastai + VAE preprocessing pipeline.
- `configs/vae_substra_skorch.toml`: Substra + skorch + VAE preprocessing pipeline.
- `configs/fedprox_non_iid.toml`: Heterogeneous data distribution benchmark.

### Running with a TOML Configuration
```powershell
# Load preset and optionally override specific parameters via CLI flags:
python main_federated_flower.py --config configs/vae_flower_fastai.toml --rounds 10
python main_federated_substra.py --config configs/vae_substra_skorch.toml --local_lr 0.0001
```

---

## 📊 Weights & Biases (W&B) Telemetry

### Online Telemetry
```powershell
python main_federated_flower.py --wandb --project cervical-cells-vae-flower-substra --run_name flower_vae_fastai
```

Tracked Telemetry:
- **Round-by-round Global Evaluation**: Holdout Accuracy, Balanced Accuracy, Macro F1, Weighted F1, AUC, LogLoss.
- **Image Restoration Telemetry**: VAE PSNR, SSIM, RMSE, and Latent Shannon Entropy.
- **Artifact Logging**: Global model checkpoints, confusion matrices, ROC curves, and side-by-side reconstruction plots.

### Offline Resilience & Fallback
If internet access is unavailable or `--no_wandb` is passed, the tracker automatically falls back to local structured persistence:
- `results/<run_id>/metrics_summary.json`
- `results/<run_id>/rounds_history.csv`
- `results/<run_id>/confusion_matrix.png`
- `results/<run_id>/roc_curves.png`
- `results/<run_id>/vae_reconstruction_sample.png`
