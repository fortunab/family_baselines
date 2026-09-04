# fastai Vision Foundation Guide

Deep learning vision pipeline powered by **[`fastai.vision.all`](https://docs.fast.ai/)**.

---

## 🔬 Core fastai Paradigms Used

1. **`DataBlock` API**:
   - Explicit separation of input data (`ImageBlock`) and target multi-class categories (`CategoryBlock`).
   - Squished item resizing with GPU-accelerated batch affine transformations (`Rotate`, `Zoom`, `Normalize`).

2. **`vision_learner` with `timm` Support**:
   - Direct support for all modern vision backbones: ConvNeXt (`convnext_base`), Vision Transformer (`vit_base_patch16_224`), EfficientNetV2 (`efficientnet_b3`), ResNet (`resnet50d`).

3. **`fine_tune` Protocol & 1cycle Policy**:
   - Two-stage transfer learning: first trains the new classification head for `freeze_epochs` while keeping the vision backbone frozen, then unfreezes the entire network with discriminative learning rates.

4. **Tracking Callbacks**:
   - `WandbCallback` and MLflow integration logging loss and metrics automatically after each epoch.

---

## 🚀 Execution Examples

```powershell
# Run with standard config:
python main_fastai.py --config configs/fastai_convnext.ini

# Run with custom learning rate and epochs:
python main_fastai.py --config configs/fastai_vit.ini --epochs 12 --learning-rate 0.0003
```
