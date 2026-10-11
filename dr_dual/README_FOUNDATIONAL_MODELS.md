# Vision Foundation Models on Dual-Database Diabetic Retinopathy

This guide details the vision foundation models evaluated on the harmonized dual database (EyePACS/APTOS + Messidor-2/IDRiD).

---

## 1. Evaluated Vision Foundation Backbones

| Architecture | Family | Parameter Count | Key Architectural Advantage in Medical Imaging |
| :--- | :--- | :---: | :--- |
| **DenseNet-121** | Dense Densely-Connected | **7.0M** | Dense feature concatenation connects every layer directly, preserving fine microaneurysm & vessel details across scales. The clinical standard in chest X-ray and ophthalmology. |
| **ResNet-50** | Deep Residual Network | **23.5M** | Identity shortcut connections eliminate vanishing gradients in deep networks, stabilizing feature extraction. |
| **ConvNeXt-Small** | 2020s Pure ConvNet | **49.5M** | Inverted bottleneck design, $7 \times 7$ depthwise convolutions, LayerNorm instead of BatchNorm, and GELU activations. Modernizes ConvNet architecture to compete with Vision Transformers. |
| **EfficientNet-B0** | Compound Scaled MBConv | **4.0M** | Joint compound scaling of network depth, width, and resolution with Squeeze-and-Excitation channel attention. Minimal parameter footprint and ultra-low latency. |

---

## 2. Experimental Setup

- **Dual Database Input**: EyePACS screening clinics (DB1) + Messidor-2 tertiary hospital Topcon fundus cameras (DB2).
- **Resolution**: $224 \times 224$ pixels, 3-channel RGB.
- **Normalization**: Standard ImageNet mean ($[0.485, 0.456, 0.406]$) and standard deviation ($[0.229, 0.224, 0.225]$).
- **Optimization**: AdamW optimizer with Cosine Annealing learning rate schedule and gradient norm clipping.
- **Objective Function**: Multi-Class Cross-Entropy Loss with Dropout ($p=0.30$) regularization.

---

## 3. Clinical Metrics Evaluated

1. **Quadratic Weighted Kappa (QWK)** ($\kappa_w$): Ordinal clinical agreement across the 5 ICDR severity grades.
2. **5-Grade Accuracy & Macro F1**: Multi-class diagnostic balance.
3. **Referral Sensitivity & Specificity**: Detection accuracy at the referable retinopathy threshold (Grade $\ge 2$).
4. **Parameter Footprint (M) & Latency / Scan (ms)**: Edge inference efficiency for point-of-care screening deployment.

---

## 4. Running the Benchmark

```bash
python run_foundational_benchmark.py --epochs 2 --batch-size 16
```
Outputs are saved to:
- `results/metrics/dr_foundational_benchmark_summary.json`
- `results/figures/foundation_models_comparison.png`
- `results/figures/foundation_efficiency_frontier.png`
- `results/checkpoints/foundation_{arch}.pth`
