# Cervical Cytology Federated Flower & Substra Suite with VAE Preprocessing

A clinical-grade cross-silo Federated Learning suite integrating **Variational Autoencoder (VAE)** cellular micrograph reconstruction with dual federated learning platforms: **Flower (`flwr`)** and **Substra**. Evaluated across **fastai** and **skorch** deep learning frameworks, 3 federated aggregation algorithms (**FedAvg**, **FedProx**, **FedAdam**), multi-candidate **Random Best Select** seed optimization (strictly rejecting seed 42), **Weights & Biases (W&B)** MLOps tracking, and 3 dedicated environment managers (**Pixi**, **Conda**, and **venv**).

---

## 🔬 Architecture & VAE Preprocessing Overview

```
                      [Raw Cervical Micrographs]
                                  │
                                  ▼
                     ┌─────────────────────────┐
                     │ VAE Cellular Denoising  │  (ConvVAE: Enc -> z ~ N(mu, var) -> Dec;
                     │ & Latent Regularization │   ELBO Loss: Recon MSE + beta * D_KL)
                     └────────────┬────────────┘
                                  │
                                  ▼
                   [Restored Standardized Cytology]
                                  │
       ┌──────────────────────────┴──────────────────────────┐
       │                                                     │
       ▼                                                     ▼
┌─────────────────────────┐                           ┌─────────────────────────┐
│ Hospital Silo Node 01   │                           │ Hospital Silo Node 02   │
│  (Flower/Substra Client)│                           │  (Flower/Substra Client)│
│  (Local fastai/skorch)  │                           │  (Local fastai/skorch)  │
└────────────┬────────────┘                           └────────────┬────────────┘
             │                                                     │
             └──────────────────────────┬──────────────────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │ Central Coordinator Server  │
                         │ (Flower / Substra Platform) │
                         │ (FedAvg / FedProx / FedAdam)│
                         └──────────────┬──────────────┘
                                        ▼
                         ┌─────────────────────────────┐
                         │ Centralized Holdout Eval    │
                         │ (10 Clinical + 4 VAE Qual)  │
                         └─────────────────────────────┘
```

---

## 📊 Preprocessing & Performance Evaluation Metrics

### Clinical Classification Metrics
1. **Accuracy**: Overall classification accuracy.
2. **Balanced Accuracy**: Unweighted macro-average recall across all 7 diagnostic cytology classes.
3. **Macro & Weighted Precision**: Positive predictive rate.
4. **Macro & Weighted Recall / Sensitivity**: True positive dysplastic/malignant detection rate.
5. **Macro Specificity**: Benign cell identification rate.
6. **Macro & Weighted F1-Score**: Harmonic mean of precision and recall.
7. **Cohen's Kappa ($\kappa$)**: Pathologist inter-rater reliability accounting for chance.
8. **Matthews Correlation Coefficient (MCC)**: Balanced metric for uneven clinical class distributions.
9. **Macro AUC-ROC (One-vs-Rest)**: Multi-class discriminative separation power.
10. **Log-Loss (Cross-Entropy)**: Probabilistic prediction confidence calibration.

### VAE Reconstruction & Image Quality Metrics
11. **PSNR (Peak Signal-to-Noise Ratio in dB)**: Logarithmic measure of micrograph signal reconstruction fidelity.
12. **SSIM (Structural Similarity Index)**: Preservation of cellular borders, luminance, and texture.
13. **RMSE (Root Mean Squared Error)**: Absolute root-mean-square pixel-intensity error between raw and restored tiles.
14. **Shannon Distribution Entropy**: Information density and chromatin detail richness.

---

## 🚀 Quickstart

### Pixi Environment
```powershell
# Setup Pixi
.\scripts\setup_pixi.ps1

# Run Flower with VAE
pixi run test-flower-vae

# Run Substra with VAE
pixi run test-substra-vae

# Run comparative benchmark
pixi run compare
```

### Conda Environment
```powershell
conda env create -f environment.yml --name cervical_cells_vae_flower_substra
conda activate cervical_cells_vae_flower_substra

# Run Flower with fastai
python main_federated_flower.py --preprocessing vae --framework fastai

# Run Substra with skorch
python main_federated_substra.py --preprocessing vae --framework skorch
```

### Standard venv Environment
```powershell
.\scripts\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python main_federated_flower.py --preprocessing vae
```

---

## 📁 Repository Structure

```
cervical_cells_vae_flower_substra/
├── configs/
│   ├── default_flower.toml
│   ├── default_substra.toml
│   ├── vae_flower_fastai.toml
│   ├── vae_flower_skorch.toml
│   ├── vae_substra_fastai.toml
│   ├── vae_substra_skorch.toml
│   └── fedprox_non_iid.toml
├── scripts/
│   ├── setup_pixi.ps1 / .sh
│   ├── setup_conda.ps1 / .sh
│   └── setup_venv.ps1 / .sh
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   └── vae_preprocessor.py
│   ├── __init__.py
│   ├── dataset.py
│   ├── evaluator.py
│   ├── fastai_engine.py
│   ├── flower_client.py
│   ├── flower_server.py
│   ├── seed_selector.py
│   ├── skorch_engine.py
│   ├── substra_engine.py
│   ├── toml_config.py
│   └── wandb_tracker.py
├── main_federated_flower.py
├── main_federated_substra.py
├── compare_cervical_vae.py
├── run_linter.py
├── pixi.toml
├── environment.yml
├── requirements.txt
└── pyproject.toml
```
