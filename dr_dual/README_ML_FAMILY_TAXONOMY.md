# Machine Learning Families Clinical Taxonomy Benchmark
## Diabetic Retinopathy Dual-Database Evaluation (EyePACS / APTOS + Messidor-2 / IDRiD)

This module provides a rigorous, taxonomy-driven benchmark of **16 machine learning models** categorized into **6 canonical clinical families** for automated diabetic retinopathy (DR) grading and referral screening across dual heterogenous retinal databases (500 fundus scans, 5 ICDR severity stages).

---

## 1. Clinical ML Family Taxonomy & Inductive Biases

| Machine Learning Family | Constituent Algorithms | Mathematical Inductive Bias | Clinical Suitability in Ophthalmology |
|---|---|---|---|
| **1. Kernel & Margin Family** | • SVM RBF ($C=10.0$)<br>• SVM Linear (Calibrated)<br>• SVM Polynomial (Deg 3) | Maximizes structural margin in reproducing kernel Hilbert space (RKHS). Effective in high dimensions ($D=42$). | Highly robust to non-linear lesion-background contrast; preserves convex separation between mild exudates and severe proliferation. |
| **2. Bagging & Random Forest Family** | • Random Forest (150 Trees)<br>• Extra Trees (Extremely Randomized)<br>• Balanced Random Forest | Ensemble averaging over decorrelated decision trees built on bootstrap subsamples; variance reduction. | Robust to multicollinearity between GLCM texture moments and vessel density; prevents over-weighting isolated microaneurysms. |
| **3. Boosting & Optimization Family** | • Histogram Gradient Boosting<br>• Gradient Tree Boosting<br>• AdaBoost (Decision Stumps) | Sequential greedy optimization fitting weak learners to pseudo-residuals of cross-entropy / log loss. | Excels at non-linear lesion feature interactions (e.g. co-occurrence of cotton wool spots + venous beading). |
| **4. Linear & Probabilistic GLM Family** | • Logistic Regression (L2 Ridge)<br>• Logistic Regression (ElasticNet)<br>• Gaussian Naive Bayes | Convex generalized linear models optimizing regularized log-likelihood; assumes monotonic log-odds. | Directly interpretable clinical odds ratios and calibrated posterior referral probabilities; fast inference ($<0.02$ ms). |
| **5. Neural & Representation Family** | • Deep MLP (256-128-64)<br>• Wide Shallow MLP (128) | Multi-layer affine transformations with non-linear activations (ReLU); hierarchical latent representations. | Learns abstract combinatorial relationships across multi-scale vessel and lesion texture moments. |
| **6. Meta-Learning & Stacking Ensemble Family** | • Soft Voting Meta-Ensemble (Top-5)<br>• Stacking Classifier (Meta-Logistic) | Second-level meta-learner aggregating diverse probability distributions across distinct algorithm families. | Harmonizes complementary inductive biases, minimizing variance and guaranteeing near-zero referral miss rates. |

---

## 2. Clinical Evaluation Metrics

All models are evaluated on the standardized pooled test set using:
1. **Quadratic Weighted Kappa ($\kappa_w$, QWK)**: Primary clinical benchmark for ordinal grading penalties. Clinical benchmark target: $\ge 0.80$ (near-perfect agreement).
2. **5-Stage ICDR Accuracy & Balanced Accuracy**: Multi-class accuracy across Stage 0 (No DR), Stage 1 (Mild), Stage 2 (Moderate), Stage 3 (Severe), Stage 4 (Proliferative).
3. **Clinical Referral Sensitivity (Grade $\ge 2$)**: Mandatory patient safety metric (requires referral to vitreoretinal specialist). Target: $\ge 95.0\%$.
4. **Referral Specificity & Negative Predictive Value (NPV)**: Prevents over-referral burden on secondary healthcare clinics.
5. **Inference Latency per Scan (ms)**: Real-time suitability for point-of-care embedded fundus cameras.

---

## 3. Running the Taxonomy Benchmark

### Windows (PowerShell)
```powershell
# Activate virtual environment
.\venv_fastai_toml\Scripts\Activate.ps1

# Run the 16-model taxonomy benchmark
python run_ml_family_taxonomy_benchmark.py --scaler robust --seed 42
```

### Linux / Ubuntu / WSL (Bash)
```bash
# Activate virtual environment
source venv_fastai_toml/bin/activate

# Run the 16-model taxonomy benchmark
python run_ml_family_taxonomy_benchmark.py --scaler robust --seed 42
```

---

## 4. Output Artifacts & Visualizations

Running the runner script automatically generates publication-grade visualizations in `results/figures/`:
- `ml_family_taxonomy_comparison.png`: Grouped horizontal bar chart showing QWK, Accuracy, and Referral Sensitivity across all 16 models grouped by family.
- `ml_family_radar_comparison.png`: 5-axis polar radar chart mapping aggregated trade-offs (QWK, Accuracy, Referral Sensitivity, Specificity, Speed) across the 6 families.
- `ml_family_best_in_family.png`: Champion comparison highlighting the top-performing model from each family.
- `results/metrics/dr_ml_family_taxonomy_summary.json`: Complete quantitative metrics export.
