# Multi-Resolution Patch-Only Neuroimaging Benchmark Guide

This document describes the multi-scale downscaling and patch-only classification pipeline for the **Alzheimer MRI Grayscale Benchmark Suite**.

---

## 1. Motivation for Downscaling & Patch Analysis
In neuropathological dementia grading:
- Global brain scans ($128 \times 128$) contain hundreds of thousands of redundant voxels.
- Downscaling to **$32 \times 32$ pixels** preserves regional lobar architecture (ventricular dilation and lateral sulcal widening).
- Downscaling to **$8 \times 8$ pixels** isolates macro-scale bilateral symmetry and severe parenchymal volume depletion.
- Restricting analysis to **patches only** evaluates whether localized regional features (e.g. ventricular core patches vs peripheral cortical patches) provide sufficient diagnostic signal without relying on global spatial continuity.

---

## 2. Multi-Resolution Patch Hierarchies

| Resolution | Downscale Dimensions | Spatial Patch Partitioning | Patch Feature Count | Anatomical Significance |
| :--- | :--- | :--- | :--- | :--- |
| **Level 1** | **$32 \times 32$ pixels** | $4 \times 4$ spatial grid of 16 patches (each $8 \times 8$ pixels) | 96 descriptors ($16 \times 6$) | Dissects medial temporal, ventricular, and parietal cortical sectors. |
| **Level 2** | **$8 \times 8$ pixels** | $2 \times 2$ macro quadrants (4 patches of $4 \times 4$) + 64 cell tokens | 88 descriptors ($24 + 64$) | Captures macroscopic brain-to-skull ratio and hemispheric atrophy asymmetry. |
| **Level 3** | **Multi-Scale Fusion** | Concatenation of $32 \times 32$ and $8 \times 8$ patch representations | 184 descriptors | Fuses multi-resolution coarse and fine patch representations. |

---

## 3. Patch Descriptors
For each extracted patch $P$, the descriptor vector computes:
1. **Mean Luminance**: $\mu_P = \frac{1}{|P|}\sum_{(x,y) \in P} P(x, y)$
2. **Standard Deviation / Contrast**: $\sigma_P = \sqrt{\frac{1}{|P|}\sum (P(x, y) - \mu_P)^2}$
3. **Peak Intensity**: $\max_{(x,y)} P(x, y)$
4. **Valley Intensity**: $\min_{(x,y)} P(x, y)$
5. **Patch Energy**: $E_P = \sum P(x, y)^2$
6. **Local Edge Magnitude**: $|\nabla P| = \text{mean}(|\partial_y P|) + \text{mean}(|\partial_x P|)$

---

## 4. Patch Saliency Mapping
Feature importance weights from the trained ensemble are mapped back onto the 2D spatial grid:
$$S(r, c) = \sum_{k=1}^6 w_{r, c, k}$$
The resulting heatmaps reveal that central patches corresponding to the lateral ventricles and medial temporal lobes carry the highest diagnostic importance.

---

## 5. Execution Commands

### Step 1: Preprocess Downscaled Patches
```bash
python preprocess_downscaled_patches.py
```
Outputs:
- Downscaled datasets in `data/processed_32x32/` and `data/processed_8x8/`
- Diagnostic 6-panel grids in `results/patches/multiscale_patch_grid_{class}.png`

### Step 2: Run Patch-Only Classification Experiments
```bash
python run_patch_experiments.py
```
Outputs:
- Resolution comparison bar chart: `results/patches/patch_resolution_comparison.png`
- Multi-scale patch saliency heatmaps: `results/patches/multiscale_patch_saliency_{class}.png`
- Metric report: `results/metrics/patch_experiments_metrics.json`
