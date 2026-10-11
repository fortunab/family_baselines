# Integrated Dual Preprocessing: Noise2Noise + CycleGAN

This document details the sequential composition of **Noise2Noise (N2N)** blind restoration and **CycleGAN** stain harmonization for digital cervical cytology.

---

## 🔄 Two-Stage Processing Pipeline

```
[Raw Micrograph] ──▶ [1. Noise2Noise Denoising] ──▶ [2. CycleGAN Harmonization] ──▶ [Model Input]
```

### Stage 1: Noise2Noise Micro-Structural Restoration
- **Input**: Raw unenhanced micrograph directly from scanner or dataset.
- **Process**: High-frequency shot noise, sensor variance, and digitization artifacts are suppressed via the U-Net restorative architecture.
- **Output**: Cleaned cytology tile with heightened cytoplasmic boundary contrast and crisp nuclear chromatin granules.

### Stage 2: CycleGAN Macro-Color Domain Normalization
- **Input**: Restored tile from Stage 1.
- **Process**: Residual generator harmonizes inter-hospital stain variability, standardizing eosinophilic cytoplasm (pink) and basophilic nuclei (deep blue/purple) into calibrated color palettes.
- **Output**: Harmonized, standardized micrograph ready for deep learning feature extraction.

---

## 📈 Diagnostic Benefits

1. **Synergistic Artifact Removal**: CycleGAN stain mapping performs significantly better when high-frequency optical sensor noise is eliminated first by Noise2Noise.
2. **Preserved Nuclear-to-Cytoplasmic (N/C) Ratio**: Both networks enforce morphological invariance so cell size, nuclear diameter, and membrane borders remain clinically intact.
3. **Cross-Silo Robustness**: Hospital silos with varying hardware scan quality and dye formulations achieve parity in feature representations.
