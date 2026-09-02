# TensorFlow Foundation Model 2: Google EfficientNetV2-L (70/15/15 Split)

Google Brain's high-capacity vision foundation model pre-trained on ImageNet-21k, implemented natively via `tf.keras.applications.EfficientNetV2L`.

---

## 📊 Splitting & Optimization Protocol
- **70% Training Set**: Used for parameter updates.
- **15% Validation Set**: Monitored after every epoch for checkpointing.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture Highlights
- Fused-MBConv layers with progressive learning scaling.
- Native ImageNet-21k pre-trained weights for superior transferability to microscopic tissue structures.
- Optimizer: AdamW with CosineDecay learning rate schedule.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_tf_foundation

# Run EfficientNetV2-L fine-tuning
python main_tf.py --model-name efficientnetv2_l --epochs 15 --batch-size 32

# Run EfficientNetV2-M
python main_tf.py --model-name efficientnetv2_m --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation
source venv/bin/activate

python3 main_tf.py --model-name efficientnetv2_l --epochs 15 --batch-size 32
```
