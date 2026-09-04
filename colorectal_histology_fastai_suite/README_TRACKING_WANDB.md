# MLOps Experiment Tracking Guide: Weights & Biases (W&B), MLflow & TensorBoard

This repository provides seamless integration with industry-standard MLOps and experiment tracking frameworks.

---

## 1. Weights & Biases (W&B)

Reference: [W&B Data Science Experiments Management](https://wandb.ai/broutonlab/first_steps/reports/Data-Science-Experiments-Management-with-Weights-Biases---Vmlldzo2NjE3MDI)

### How to use W&B:
1. Log in with your W&B API key:
   ```bash
   wandb login
   ```
2. Enable W&B in configuration or via CLI:
   ```powershell
   python main_fastai.py --config configs/fastai_convnext.ini --tracking-backend wandb
   ```
3. If running offline without credentials, the tracker automatically falls back to safe offline logging mode (`WANDB_MODE=offline`).

---

## 2. MLflow

Reference: [MLflow Documentation](https://mlflow.org/)

### How to use MLflow:
```powershell
python main_fastai.py --config configs/fastai_convnext.ini --tracking-backend mlflow
```
Launch the MLflow UI to view runs:
```bash
mlflow ui
```

---

## 3. TensorBoard

### How to use TensorBoard:
```powershell
python main_fastai.py --config configs/fastai_convnext.ini --tracking-backend tensorboard
```
Launch TensorBoard:
```bash
tensorboard --logdir results/tensorboard_logs
```

---

## 4. Local / Offline Telemetry Fallback

When `--tracking-backend offline` is selected (or when running without internet access), all experiment parameters, epoch metrics, confusion matrices, and ROC curves are saved locally to:
- `results/telemetry_<experiment_name>.json`
- `results/telemetry_<experiment_name>.csv`
- `results/metrics_<model>.json`
