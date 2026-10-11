# Variational Autoencoder (VAE) Micrograph Preprocessing

Variational Autoencoders (Kingma & Welling, 2013) provide a principled probabilistic framework for image reconstruction and latent representation modeling.

---

## 🔬 Clinical Motivation for Digital Cytology

In Pap smear slide digitization:
1. Micrographs suffer from CCD/CMOS sensor shot noise, focal plane defocus, and illumination variance across scanning batches.
2. Standard autoencoders can overfit high-frequency noise artifacts, whereas VAEs enforce a smooth Gaussian prior $\mathcal{N}(0, I)$ over the latent space.
3. The ELBO loss functions as a biological denoiser: the encoder extracts salient morphological chromatin patterns ($\mu, \sigma$), while the decoder reconstructs a regularized, artifact-free cytology tile.

---

## 📐 Mathematical Formulation

### Reparameterization Trick
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\epsilon} \odot \boldsymbol{\sigma}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \mathbf{I})$$

### Evidence Lower Bound (ELBO) Loss
$$\mathcal{L}_{\text{ELBO}}(\mathbf{x}; \theta, \phi) = \mathbb{E}_{q_\phi(\mathbf{z}|\mathbf{x})}[\log p_\theta(\mathbf{x}|\mathbf{z})] - \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z}))$$

- **Reconstruction Loss**: Mean Squared Error (MSE) between original tile and reconstructed micrograph.
- **Kullback-Leibler Divergence**:
  $$D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}) \parallel p(\mathbf{z})) = -\frac{1}{2} \sum_{j=1}^J \left(1 + \log(\sigma_j^2) - \mu_j^2 - \sigma_j^2\right)$$
- **$\beta$ Scaling**: Controls balance between fidelity and regularized smooth representation.

---

## 🏗️ Architecture

- **Encoder**: 4 convolutional downsampling stages with Batch Normalization and LeakyReLU(0.2), followed by linear projection to latent mean ($\boldsymbol{\mu}$) and log-variance ($\log \boldsymbol{\sigma}^2$) of dimension $d=128$.
- **Decoder**: Linear projection expanding latent $z$ into spatial feature maps, followed by 4 transpose convolutional stages and Sigmoid activation.
- **Residual Blend**: $0.85 \times \text{reconstructed} + 0.15 \times \text{original}$, preserving true chromatin granulometry while suppressing scanner artifacts.
