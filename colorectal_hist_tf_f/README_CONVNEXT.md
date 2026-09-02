# TensorFlow Foundation Model 1: Meta ConvNeXt-Large (70/15/15 Split)

Meta's pure modern convolutional foundation model implemented natively in TensorFlow (`tf.keras.applications.ConvNeXtLarge`) for 8-class colorectal tissue histology.

---

## 📊 Splitting & Optimization Protocol
- **70% Training Set**: Used for backpropagation parameter updates.
- **15% Validation Set**: Monitored after every epoch for checkpointing `best_tf_model.weights.h5`.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture Highlights
- $7 \times 7$ Depthwise separable convolutions with inverted bottlenecks and Layer Normalization.
- Pathology Augmentations: Random 90-degree rotations, horizontal/vertical flips, contrast & brightness jitter.
- Optimizer: AdamW with CosineDecay learning rate schedule.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_tf_foundation

# Run ConvNeXt-Large fine-tuning
python main_tf.py --model-name convnext_large --epochs 15 --batch-size 32

# Run ConvNeXt-Base
python main_tf.py --model-name convnext_base --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation
source venv/bin/activate

python3 main_tf.py --model-name convnext_large --epochs 15 --batch-size 32
```
