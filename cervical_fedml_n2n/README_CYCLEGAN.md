# CycleGAN Unpaired Stain Harmonization

Cycle-Consistent Generative Adversarial Networks (Zhu et al., 2017) facilitate image-to-image translation between unpaired image distributions.

---

## 🔬 Clinical Motivation for Decentralized Hospital Silos

In multi-center digital pathology and cervical cytology:
- Hospital Silo A may use Hematoxylin & Orange G-6 stains from Supplier X.
- Hospital Silo B may use EA-50 formulations with differing incubation times and illumination spectra.
- This creates severe **inter-site domain shift** (stain batch effects), which degrades the generalization of federated models across silos.

CycleGAN harmonizes heterogeneous stain distributions to a reference Papanicolaou standard without altering cellular geometry:
$$\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(X)} [\|F(G(\mathbf{x})) - \mathbf{x}\|_1] + \mathbb{E}_{\mathbf{y} \sim p_{\text{data}}(Y)} [\|G(F(\mathbf{y})) - \mathbf{y}\|_1]$$

---

## 🏗️ Generator Architecture

- Downsampling: $7 \times 7$ reflection-padded convolutions followed by stride-2 downsampling.
- Residual Core: 6 ResNet blocks with instance normalization.
- Upsampling: ConvTranspose2d with reflection padding and Tanh activation.
- Color Transform: LAB color space alignment anchoring cellular luminance while mapping chrominance distributions to standard cytology references.
