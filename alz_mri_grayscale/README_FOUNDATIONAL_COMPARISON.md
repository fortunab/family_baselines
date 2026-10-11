# Vision Foundation Backbones: Alzheimer MRI Comparison Guide

Reference guide for evaluating Vision Foundation Models on grayscaled Alzheimer MRI brain scans.

---

## 1. Selected Vision Foundation Backbones

| Model | Paradigm | Parameters | Key Radiological Advantage |
| :--- | :--- | :--- | :--- |
| **DenseNet-121** | Dense Feature Reuse | ~8.0 M | Direct feature concatenation across all layers preserves fine-grained hippocampal contours. |
| **ResNet-50** | Deep Residual Learning | ~25.6 M | Bottleneck residual blocks prevent gradient vanishing across deep feature hierarchies. |
| **ConvNeXt-Small** | 2020s Modernized ConvNet | ~50.2 M | 7x7 depthwise convolutions and LayerNorm provide Transformer-scale receptive field. |
| **EfficientNet-B0** | Compound Scaling | ~5.3 M | Inverted MBConv blocks with Squeeze-and-Excitation (SE) channel attention optimize parameters. |

---

## 2. Training & Fine-Tuning Setup
- **Framework**: fastai `vision_learner` with 1-Cycle learning rate policy.
- **Input**: Preprocessed single-channel grayscale MRI scans replicated to 3 channels.
- **Dimensions**: $224 \times 224$ pixels.
- **Batch Size**: 16.
- **Learning Rate**: $10^{-3}$ ($5 \times 10^{-4}$ for ConvNeXt).

---

## 3. Comparison Dimensions Against Classical ML Family
1. **Diagnostic Accuracy & Balanced Accuracy**: Direct classification correctness.
2. **Clinical Sensitivity & Specificity**: Macro recall and specificity across all 4 stages.
3. **Inference Latency**: Milliseconds per slice on CPU vs GPU.
4. **Computational Footprint**: Parameter count and model size on disk.

---

## 4. Execution
```bash
python run_foundational_comparison.py --epochs 2 --batch-size 16
```

Outputs:
- Metric report: `results/metrics/foundational_comparison_metrics.json`
