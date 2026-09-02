# TensorFlow Foundation Model 4: Big Transfer BiT / ResNet152V2 (70/15/15 Split)

Google Brain's Big Transfer (BiT) representation foundation model pre-trained on large-scale data, implemented via `tf.keras.applications.ResNet152V2`.

---

## 📊 Splitting & Optimization Protocol
- **70% Training Set**: Used for parameter updates.
- **15% Validation Set**: Monitored after every epoch for checkpointing.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture Highlights
- 152-layer deep residual architecture with pre-activation bottlenecks.
- Group/Batch normalization for stable transfer learning on medical histology tiles.
- Optimizer: AdamW with CosineDecay learning rate schedule.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_tf_foundation

# Run BiT / ResNet152V2 fine-tuning
python main_tf.py --model-name bit_resnet152v2 --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\colorectal_histology_tf_foundation
source venv/bin/activate

python3 main_tf.py --model-name bit_resnet152v2 --epochs 15 --batch-size 32
```
