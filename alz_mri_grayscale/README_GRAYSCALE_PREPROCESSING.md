# Grayscale Preprocessing Pipeline for Alzheimer Brain MRI

Clinical neuroimaging image enhancement reference for the **Alzheimer MRI Grayscale Benchmark Suite**.

---

## 1. Neuroimaging Rationale
Raw brain MRI scans often exhibit:
- Scanner-specific coil bias fields and high-frequency thermal acquisition noise.
- Non-standardized multi-channel (pseudo-RGB) formats that dilute radiological luminance.
- Poor contrast between cerebral cortex gray matter and subcortical white matter.
- Sulcal and ventricular boundaries obscured by low dynamic range.

The grayscale enhancement pipeline resolves these issues through 5 standardized stages.

---

## 2. Mathematical Pipeline Formulation

### Stage 1: Single-Channel Luminance Conversion
Converts multi-channel representations to scalar luminance using ITU-R BT.601 standard weights:
$$Y = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B$$

### Stage 2: 2D Gaussian Spatial Noise Filtration
Applies an isotropic Gaussian kernel ($\sigma = 0.8$) to suppress thermal sensor grain without blurring fine sulcal edges:
$$G(x, y) = \frac{1}{2\pi \sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}}$$
$$I_{\text{denoised}}(x, y) = I(x, y) * G(x, y)$$

### Stage 3: Contrast-Limited Adaptive Histogram Equalization (CLAHE)
Partitions the 2D scan into an $8 \times 8$ grid of local contextual tiles:
1. Calculates local histogram $H_k(i)$ for tile $k$.
2. Clips histogram bins at clip limit $T = \text{clip\_limit} \cdot \frac{N_{\text{tile}}}{N_{\text{bins}}}$.
3. Redistributes clipped excess uniformly across all $N_{\text{bins}} = 256$ bins.
4. Computes normalized Cumulative Distribution Function $\text{CDF}_k(i)$.
5. Bilinearly interpolates transfer functions across tile centers to avoid boundary artifacts:
   $$I_{\text{CLAHE}}(x, y) = \sum_{k \in \mathcal{N}} w_k(x, y) \cdot \text{CDF}_k(I_{\text{denoised}}(x, y))$$

### Stage 4: Parenchymal Skull-Strip & Background Masking
Detects brain parenchyma by thresholding out the background scanner air ($T = 0.04$):
$$M(x, y) = \mathbb{I}(I_{\text{norm}}(x, y) > 0.04)$$

### Stage 5: Dynamic Range Percentile Normalization
Clips outliers at the $1^{\text{st}}$ and $99^{\text{th}}$ percentiles to eliminate extreme artifacts:
$$I_{\text{final}}(x, y) = \text{clip}\left(\frac{I_{\text{CLAHE}}(x, y) - P_1}{P_{99} - P_1}, 0.0, 1.0\right) \cdot M(x, y)$$

---

## 3. Preprocessing Execution
```bash
python preprocess_grayscale_dataset.py
```
Outputs:
- Enhanced 8-bit single-channel slices in `data/processed_grayscale/{class}/`
- 5-panel diagnostic figures in `results/preprocessing/preproc_breakdown_{class}.png`
