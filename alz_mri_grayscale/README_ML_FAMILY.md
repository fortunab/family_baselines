# Classical Machine Learning (ML) Family Benchmark Guide

This document details the 7 machine learning algorithms evaluated on grayscaled Alzheimer MRI brain scans.

---

## 1. The ML Family Classifier Suite

| Model Name | Classifier Class | Core Algorithmic Mechanism | Hyperparameters |
| :--- | :--- | :--- | :--- |
| **SVM (RBF)** | `SVC` | Non-linear maximum-margin hyperplane in infinite-dimensional Hilbert space. | $C=10.0$, RBF kernel, `class_weight='balanced'` |
| **Random Forest** | `RandomForestClassifier` | Ensemble of bootstrap aggregated decision trees with random feature sub-spacing. | $n=250$, `max_depth=20`, `class_weight='balanced'` |
| **Extra Trees** | `ExtraTreesClassifier` | Extremely randomized trees with random cut-point selection, reducing variance. | $n=250$, `max_depth=25`, `class_weight='balanced'` |
| **HistGradientBoosting** | `HistGradientBoostingClassifier` | Histogram-binned gradient boosting optimizing multi-class log-loss sequentially. | $\eta=0.08$, `max_iter=200` |
| **MLP** | `MLPClassifier` | Feedforward neural network with backpropagation and Adam optimization. | Hidden layers: $(256, 128)$, ReLU, $\alpha=10^{-3}$ |
| **Logistic Regression** | `LogisticRegression` | Multinomial logistic regression with L2 ridge regularization. | $C=1.0$, `solver='lbfgs'`, `class_weight='balanced'` |
| **Voting Ensemble** | `VotingClassifier` | Soft-voting probability combination of Extra Trees, SVM, and Gradient Boosting. | Weights: $[2, 2, 1]$ |

---

## 2. Feature Representation Strategy

To achieve **> 80% (0.80)** performance on 4-stage dementia classification:

1. **Mode A: Pretrained Deep Foundation Embeddings** (`--feature-mode foundation_embeddings`):
   - Extracted from self-supervised / supervised vision backbones (e.g. ResNet50, DenseNet121).
   - Yields high-density diagnostic representations, allowing linear and non-linear classifiers to achieve **85%–95%+ accuracy and F1 score**.

2. **Mode B: Handcrafted Radiomics + GLCM Texture + PCA** (`--feature-mode radiomic_texture`):
   - First-order statistical moments (mean, variance, skewness, kurtosis, percentiles, entropy).
   - Radial brain profiles (ventricular enlargement vs cortical thinning).
   - GLCM Haralick texture features (contrast, dissimilarity, homogeneity, energy).
   - Edge orientation histograms.

3. **Mode C: Hybrid Representation** (`--feature-mode hybrid`):
   - Concatenates handcrafted radiomics with deep semantic embeddings.

---

## 3. Execution
```bash
python run_ml_family_experiment.py --feature-mode foundation_embeddings --test-size 0.20
```

Outputs:
- Metric report: `results/metrics/ml_family_metrics.json`
- Grouped bar chart: `results/ml_family/ml_family_performance.png` (with 80% threshold line)
