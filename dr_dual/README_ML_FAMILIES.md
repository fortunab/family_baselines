# Machine Learning Family Reference & Algorithm Selection

This guide details the mathematical foundations and hyperparameters for each of the 9 machine learning algorithms evaluated in this suite.

---

## 1. Algorithms & Clinical Strengths

### 1. Support Vector Classifier (RBF Kernel)
- **Kernel**: Radial Basis Function $K(x, x') = \exp(-\gamma ||x - x'||^2)$.
- **Strength**: Projects radiomic biomarker vectors into infinite-dimensional Hilbert space, separating non-linear lesion grades cleanly.
- **Hyperparameters**: $C=10.0, \gamma=\text{'scale'}$, Platt scaling probability calibration.

### 2. Support Vector Classifier (Linear Kernel)
- **Formulation**: Hyperplane margin maximization with calibrated probabilities via 3-fold sigmoid calibration.
- **Strength**: Explicit feature weights directly reveal which radiomic biomarkers drive grading decisions.

### 3. Random Forest Classifier
- **Architecture**: 150 bagged decision trees with randomized feature subsampling ($\sqrt{D}$).
- **Strength**: Highly resilient to collinear radiomic features (e.g. correlated color moments).

### 4. Extra Trees Classifier (Extremely Randomized Trees)
- **Architecture**: 150 trees with cut-point randomization.
- **Strength**: Further variance reduction compared to Random Forest; smoother decision boundaries for subtle lesion stages.

### 5. Histogram Gradient Boosting Classifier (HistGBM)
- **Architecture**: LightGBM-style gradient boosted decision trees binning continuous radiomic features into 256 discrete integer bins.
- **Strength**: Fast training, automatic non-linear feature interaction learning.

### 6. Multi-Layer Perceptron (MLP)
- **Architecture**: Fully-connected feedforward network with two hidden layers: $(128, 64)$ with ReLU activations and L2 weight decay ($\alpha=0.001$).
- **Strength**: Learns composite biomarker representations.

### 7. Multinomial Logistic Regression
- **Formulation**: Softmax cross-entropy minimization with L2 regularization ($C=2.0$).
- **Strength**: Clinical posterior probabilities are directly interpretable as risk odds ratios.

### 8. K-Nearest Neighbors (KNN)
- **Distance Metric**: Inverse distance-weighted Euclidean metric in normalized radiomic feature space ($k=5$).
- **Strength**: Instance-based analog reasoning ("this fundus is most similar to these 5 historical patient scans").

### 9. Soft Voting Ensemble
- **Meta-Architecture**: Weighted consensus model aggregating predicted probability distributions across SVM RBF, Random Forest, Extra Trees, HistGBM, and MLP:
  $$\hat{P}(y = c | x) = \frac{1}{M} \sum_{m=1}^M P_m(y = c | x)$$
- **Strength**: Consistently achieves the highest Quadratic Weighted Kappa ($\kappa_w$) by canceling individual model variance.
