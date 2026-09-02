# TensorFlow Colorectal Histology Foundation Models Suite (70/15/15 Split)

This repository provides reproducible **TensorFlow 2.x / Keras** implementations of **4 performant Foundation Models** adapted for the **Colorectal Histology Benchmark** (5,000 $150 \times 150$ H&E tiles across 8 tissue classes):

---

## 🏆 The 4 TensorFlow Foundation Models

| Model # | Foundation Architecture | Pre-training Scale | TF / Keras Implementation | Detailed Guide |
| :---: | :--- | :--- | :--- | :--- |
| **1** | **Meta ConvNeXt-Large / Base** | ImageNet-22k / 1k (Modern ConvNet) | `tf.keras.applications.ConvNeXtLarge` | [README_CONVNEXT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation/README_CONVNEXT.md) |
| **2** | **Google EfficientNetV2-L** | ImageNet-21k (Fused-MBConv SOTA) | `tf.keras.applications.EfficientNetV2L` | [README_EFFICIENTNET.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation/README_EFFICIENTNET.md) |
| **3** | **Google Vision Transformer (ViT-Base)** | ImageNet-21k (14M Images) | `google/vit-base-patch16-224-in21k` | [README_VIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation/README_VIT.md) |
| **4** | **Google Big Transfer (BiT / ResNet152V2)** | ImageNet-21k / JFT Representation | `tf.keras.applications.ResNet152V2` | [README_BIT.md](file:///C:/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation/README_BIT.md) |

---

## 🔬 Dataset & Tissue Classes

```text
colorectal_histology_tf_foundation/data/
├── 01_TUMOR/      # Colorectal Adenocarcinoma Epithelium (TUM)
├── 02_STROMA/     # Cancer-Associated Stroma (STR)
├── 03_COMPLEX/    # Complex Stroma / Mixed Glands (COMP)
├── 04_LYMPHO/     # Immune Cells / Lymphocytes (LYM)
├── 05_DEBRIS/     # Necrotic Debris & Mucus (DEB)
├── 06_MUCOSA/     # Normal Colon Mucosa (NORM)
├── 07_ADIPOSE/    # Adipose Fat Tissue (ADI)
└── 08_EMPTY/      # Background / Glass Slide (BACK)
```

---

## 📊 Splitting & Random Seed Protocol
- **70% Training Set**: Used for parameter updates via AdamW + CosineDecay learning rate schedule.
- **15% Validation Set**: Monitored after every epoch for checkpointing the best model weights.
- **15% Test Set**: Evaluated once on the best saved checkpoint for true holdout metrics.
- **Dynamic Random Seed**: Automatically drawn from system entropy for each run.

---

## 🚀 Quick Start & How to Run

### 🪟 Windows (PowerShell / Command Prompt)

```powershell
# Navigate to the project directory
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_tf_foundation

# 1. Run Model 1: ConvNeXt-Large (70/15/15 Split)
python main_tf.py --model-name convnext_large --epochs 15 --batch-size 32

# 2. Run Model 2: EfficientNetV2-L (70/15/15 Split)
python main_tf.py --model-name efficientnetv2_l --epochs 15 --batch-size 32

# 3. Run Model 3: Vision Transformer ViT-Base (70/15/15 Split)
python main_tf.py --model-name vit_base --epochs 15 --batch-size 32

# 4. Run Model 4: Big Transfer BiT / ResNet152V2 (70/15/15 Split)
python main_tf.py --model-name bit_resnet152v2 --epochs 15 --batch-size 32

# Or sequentially run all 4 models:
python main_tf.py --model-name all --epochs 15

# Compare all 4 Foundation Models side-by-side:
python tf_compare.py
```

---

### 🐧 Ubuntu / WSL (Linux)

```bash
# Navigate to the project directory
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation

# Activate virtual environment
source venv/bin/activate

# 1. ConvNeXt-Large
python3 main_tf.py --model-name convnext_large --epochs 15 --batch-size 32

# 2. EfficientNetV2-L
python3 main_tf.py --model-name efficientnetv2_l --epochs 15 --batch-size 32

# 3. Vision Transformer ViT-Base
python3 main_tf.py --model-name vit_base --epochs 15 --batch-size 32

# 4. Big Transfer BiT / ResNet152V2
python3 main_tf.py --model-name bit_resnet152v2 --epochs 15 --batch-size 32

# 4-Model Comparison Leaderboard & Chart
python3 tf_compare.py
```
