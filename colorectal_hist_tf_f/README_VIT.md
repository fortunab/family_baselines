# TensorFlow Foundation Model 3: Google Vision Transformer ViT-Base (70/15/15 Split)

Pure multi-head self-attention Vision Transformer pre-trained on ImageNet-21k (`google/vit-base-patch16-224-in21k`) implemented in TensorFlow / Keras for 8-class colorectal tissue histology.

---

## 📊 Splitting & Optimization Protocol
- **70% Training Set**: Used for parameter updates.
- **15% Validation Set**: Monitored after every epoch for checkpointing.
- **15% Test Set**: Evaluated once on the best saved checkpoint.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🔬 Architecture Highlights
- 12 Transformer encoder blocks with 12 self-attention heads ($16\times 16$ patch tokens).
- Pre-trained on 14 million images from ImageNet-21k.
- Optimizer: AdamW with CosineDecay learning rate schedule.

---

## 🚀 How to Run

### 🪟 Windows (PowerShell / CMD)
```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_tf_foundation

# Run ViT-Base fine-tuning
python main_tf.py --model-name vit_base --epochs 15 --batch-size 32
```

### 🐧 Ubuntu / WSL (Linux)
```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity\scratch\colorectal_histology_tf_foundation
source venv/bin/activate

python3 main_tf.py --model-name vit_base --epochs 15 --batch-size 32
```
