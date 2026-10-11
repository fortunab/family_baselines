# Class Activation Mapping (CAM) Explainability Guide for Alzheimer MRI

Comprehensive mathematical and algorithmic reference for the three explainable AI methods implemented in the **Alzheimer MRI fastai CAM Suite**:
1. **Grad-CAM** (Selvaraju et al., ICCV 2017)
2. **Grad-CAM++** (Chattopadhay et al., WACV 2018)
3. **Score-CAM** (Wang et al., CVPRW 2020)
4. **Tri-CAM Consensus & Disagreement / Variance Mapping**

---

## 1. Motivation in Neuroimaging
In clinical neuropathology, diagnosing Alzheimer's Disease involves detecting localized biomarkers:
- **Hippocampal Atrophy**: Shrinkage of medial temporal lobe structures.
- **Ventricular Enlargement**: Dilation of the lateral ventricles due to parenchymal volume loss.
- **Cortical Thinning**: Sulcal widening across parietal and temporal lobes.

Standard black-box convolutional networks can overfit to skull boundaries or imaging artifacts. CAM algorithms verify whether the deep model attends to true neurodegenerative anatomy.

---

## 2. Grad-CAM (1st-Order Gradient Backpropagation)

### Mathematical Formulation
Grad-CAM computes the importance weight $\alpha_k^c$ for each feature activation map $A^k$ corresponding to diagnostic class $c$ via global average pooling of first-order partial derivatives:

$$\alpha_k^c = \frac{1}{Z} \sum_{i} \sum_{j} \frac{\partial Y^c}{\partial A_{i,j}^k}$$

where:
- $Y^c$ is the unnormalized score (logit) for class $c$.
- $A_{i,j}^k$ is the activation at spatial coordinates $(i, j)$ in feature map $k$.
- $Z = H \times W$ is the spatial dimensions of the feature map.

The final heat map $L_{\text{Grad-CAM}}^c$ is computed as a weighted linear combination followed by a Rectified Linear Unit (ReLU) to isolate positive contributions:

$$L_{\text{Grad-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

---

## 3. Grad-CAM++ (Generalized Higher-Order Gradient Weighting)

### Mathematical Formulation
Standard Grad-CAM struggles when multiple distinct pathological occurrences exist across the slice (e.g. bilateral ventricular enlargement) or when gradients vanish across large regions. Grad-CAM++ uses higher-order (second and third) derivatives to compute pixel-specific weights $w_k^c$:

$$w_k^c = \sum_{i} \sum_{j} \alpha_{i,j}^{k,c} \cdot \text{ReLU}\left( \frac{\partial Y^c}{\partial A_{i,j}^k} \right)$$

where the spatial weighting factor $\alpha_{i,j}^{k,c}$ is defined analytically as:

$$\alpha_{i,j}^{k,c} = \frac{\frac{\partial^2 Y^c}{\partial (A_{i,j}^k)^2}}{2 \frac{\partial^2 Y^c}{\partial (A_{i,j}^k)^2} + \sum_{a,b} A_{a,b}^k \frac{\partial^3 Y^c}{\partial (A_{a,b}^k)^3}}$$

The resulting attribution map provides sharper boundaries around fine anatomical contours like the subiculum and hippocampal fissure.

---

## 4. Score-CAM (Gradient-Free Perturbation Weighting)

### Mathematical Formulation
Gradient-based CAMs can suffer from gradient saturation and noise. **Score-CAM** operates completely independent of backpropagated gradients:

1. **Activation Normalization**: Each feature map $A^k$ is upsampled to the input image dimensions $(H, W)$ and normalized into $[0, 1]$:
   $$M^k = \frac{\text{Upsample}(A^k) - \min(A^k)}{\max(A^k) - \min(A^k) + \epsilon}$$

2. **Phase Masking**: The normalized activation acts as a soft spatial mask on the raw brain MRI slice $X$:
   $$X_k = X \odot M^k$$

3. **Forward-Pass Scoring**: The masked image is forwarded through the network, obtaining the target class score:
   $$\alpha_k^c = C(X_k) = f(X \odot M^k)_c - f(X_b)_c$$
   where $X_b$ is a baseline uninformative image.

4. **Linear Combination**:
   $$L_{\text{Score-CAM}}^c = \text{ReLU}\left( \sum_{k} \alpha_k^c A^k \right)$$

To ensure sub-second inference on MRI scans, our suite features **energy channel pruning** (selecting the top 32 highest-variance feature channels) and batched forward passes.

---

## 5. Tri-CAM Consensus & Disagreement / Variance Mapping

To eliminate single-algorithm bias, the suite synthesizes all three techniques:

### Consensus Map
Measures the voxel-level clinical agreement:
$$\bar{M}_{\text{consensus}}(i, j) = \frac{1}{3} \left( M_{\text{Grad-CAM}}(i, j) + M_{\text{Grad-CAM++}}(i, j) + M_{\text{Score-CAM}}(i, j) \right)$$

### Inter-Method Variance / Disagreement Map
Identifies areas where methods diverge (e.g., gradient noise vs actual feature contribution):
$$\sigma^2(i, j) = \frac{1}{3} \sum_{m \in \{\text{Grad}, \text{Grad++}, \text{Score}\}} \left( M_m(i, j) - \bar{M}_{\text{consensus}}(i, j) \right)^2$$
Regions with low variance and high consensus provide high diagnostic confidence for clinical interpretation.
