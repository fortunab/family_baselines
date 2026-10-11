# Noise2Noise (N2N) Micrograph Denoising

Noise2Noise (Lehtinen et al., 2018) is a self-supervised deep image restoration framework capable of removing high-frequency sensor and optical noise **without requiring clean ground-truth scans**.

---

## 🔬 Clinical Motivation for Pap Smears

In digital cervical cytology:
1. Low-light optical sensors and CCD/CMOS slide scanners introduce Poisson-Gaussian shot noise.
2. Fine nuclear chromatin textures (vital for distinguishing low-grade SIL from high-grade carcinoma in situ) are frequently obscured by optical granularity.
3. Conventional denoising methods (such as Gaussian blurring or bilateral filtering) soften critical nuclear borders and compromise diagnostic granularity.

Noise2Noise maps zero-mean noisy inputs to restored predictions:
$$\mathbb{E}[\mathbf{y}_2 \mid \mathbf{y}_1] = \mathbf{x}_{\text{clean}}$$
where $\mathbf{y}_1, \mathbf{y}_2$ are independent noisy observations of the same underlying specimen $\mathbf{x}_{\text{clean}}$.

---

## 🏗️ Architecture

- Architecture: Deep U-Net with skip connections.
- Encoder: 3 downsampling blocks with LeakyReLU activations and MaxPool layers.
- Bottleneck: Deep convolutional feature extraction preserving high-dimensional cellular structures.
- Decoder: 3 upsampling blocks with transpose convolutions concatenating encoder skip features.
- Output: Residual addition with clamp $[0, 1]$, suppressing background sensor variance while preserving cell chromatin.
