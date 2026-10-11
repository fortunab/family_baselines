# TOML Configuration & Weights & Biases (W&B) Telemetry

This suite integrates hierarchical TOML configuration with real-time Weights & Biases MLOps tracking.

---

## ⚙️ TOML Architecture

Configs are organized into structured sections:
- `[general]`: Random seed, device selection, output directory, backend framework.
- `[preprocessing]`: Preprocessing mode (`none`, `n2n`, `cyclegan`, `n2n_cyclegan`).
- `[dataset]`: Herlev dataset parameters, non-IID Dirichlet distribution ($\alpha$), holdout test split (15%).
- `[model]`: Vision backbone (`convnext_small`), pretraining flags, dropout rate.
- `[fedml]`: Strategy (`FedAvg`, `FedProx`, `FedAdam`), communication rounds, client count, local epochs, learning rate.
- `[wandb]`: Project name, run name, and tracking mode (`online` or `offline`).

---

## 📊 W&B Artifacts & Exports

During and after training, the following artifacts are generated:
- Per-round clinical metrics (Loss, Accuracy, Balanced Acc, F1-Macro, Precision, Recall, Specificity, Kappa, MCC, AUC).
- Micrograph restoration metrics (PSNR dB, SSIM, Entropy).
- Diagnostic plots: Confusion matrices, multi-class ROC curves, 4-panel visual comparison.
- Local persistence: `results/fedml_*_metrics_history.json` and `.csv`.
