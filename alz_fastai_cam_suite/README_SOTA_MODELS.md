# SOTA Multi-Model Neuroimaging Benchmark Guide

This document describes the 4 State-of-the-Art (SOTA) deep learning architectures evaluated in the **Alzheimer MRI fastai CAM Explainability Suite**.

---

## 1. Selected SOTA Architectures

| Architecture | Paradigm | Parameters | Key Neuroimaging Strength | Feature Hook Layer for CAM |
| :--- | :--- | :--- | :--- | :--- |
| **ResNet-50** | Deep Residual Network | ~25.6 M | Bottleneck residual learning prevents gradient vanishing across 50 layers; standard neuroimaging baseline. | `encoder.layer4[-1]` |
| **DenseNet-121** | Dense Feature Reuse | ~8.0 M | Direct feature concatenation across all layers preserves fine hippocampal boundaries and sulcal edges. | `encoder.features[-1]` |
| **EfficientNet-B0** | Compound Scaling | ~5.3 M | Inverted MBConv blocks with Squeeze-and-Excitation (SE) channel attention highlight salient atrophy foci. | `encoder.features[-1]` |
| **ConvNeXt-Small** | 2020s Modern ConvNet | ~50.2 M | 7x7 depthwise convolutions and LayerNorm provide Transformer-grade receptive field with convolutional inductive bias. | `encoder.stages[-1]` |

---

## 2. Multi-Model Experiment Execution

### Running All 4 SOTA Models in One Command
To sequentially train, evaluate Step 1 clinical metrics, generate Tri-CAM heatmaps (Step 2), and build the leaderboard:
```bash
python run_sota_multi_model_experiments.py --epochs 2 --batch-size 16
```

### Running an Individual SOTA Architecture
```bash
# Example: Evaluate DenseNet-121 only
python visualize_cam_heatmaps.py --arch densenet121

# Example: Evaluate ConvNeXt-Small only
python visualize_cam_heatmaps.py --arch convnext_small

# Example: Evaluate EfficientNet-B0 only
python visualize_cam_heatmaps.py --arch efficientnet_b0

# Example: Evaluate ResNet-50 only
python visualize_cam_heatmaps.py --arch resnet50
```

---

## 3. Evaluated Metrics for Each Architecture

### Step 1: 10 Clinical Classification Metrics
- Overall Accuracy & Balanced Accuracy
- Macro Precision & Weighted Precision
- Macro Recall (Sensitivity) & Weighted Recall
- Macro Specificity (True Negative Rate)
- Macro F1-Score & Weighted F1-Score
- Cohen's Kappa ($\kappa$)
- Matthews Correlation Coefficient (MCC)
- ROC-AUC Macro (One-vs-Rest)
- Multiclass Cross-Entropy Log-Loss
- Confusion Matrix Heatmap

### Step 2: 5 CAM Explainability & Attribution Metrics
- **Average Drop (%) on Masking**: Faithfulness of CAM-highlighted regions
- **Ablation Drop (%)**: Performance reduction when salient regions are removed
- **Increase in Confidence (%)**: Cases where masking removes background confounders
- **Brain Energy Localization Ratio**: Percentage of CAM activation within true brain tissue vs background air
- **Attribution Shannon Entropy**: Spatial concentration vs dispersion of attention
- **Inter-CAM Consensus SSIM & Correlation**: Structural similarity across Grad-CAM, Grad-CAM++, and Score-CAM

---

## 4. Generated Artifacts & Comparison Matrix

Running the multi-model experiments automatically creates:
- `results/heatmaps/multi_model_benchmark_matrix.png`: 4-panel comparison figure across all models
- `results/metrics/multi_model_benchmark_summary.json`: Aggregated metrics table
- Per-model subdirectories:
  - `results/heatmaps/{model}/cam_breakdown_{class}.png`
  - `results/heatmaps/{model}/dementia_progression_gallery.png`
  - `results/heatmaps/{model}/cam_metrics_comparison.png`
  - `results/heatmaps/{model}/classification_confusion_matrix.png`
  - `results/metrics/{model}/step_metrics_summary.json`
  - `results/models/alzheimer_sota_{model}.pth`
