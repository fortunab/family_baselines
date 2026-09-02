# Running TensorFlow Colorectal Histology Baselines on Windows (PowerShell & CMD)

This guide walks you through setting up and running all 4 TensorFlow Foundation Models on the **Colorectal Histology dataset** in **Windows 10/11** using the **70% Train / 15% Val / 15% Test** evaluation protocol.

---

## 1. Environment & Setup

Open **PowerShell** or **Command Prompt** and navigate to the project directory:

```powershell
cd C:\Users\Lenovo\.gemini\antigravity\scratch\colorectal_histology_tf_foundation

# Install dependencies
pip install -r requirements.txt
```

---

## 2. Running Individual Foundation Models

### Model 1: Meta ConvNeXt-Large / Base
```powershell
python main_tf.py --model-name convnext_large --epochs 15 --batch-size 32
```

### Model 2: Google EfficientNetV2-L
```powershell
python main_tf.py --model-name efficientnetv2_l --epochs 15 --batch-size 32
```

### Model 3: Google Vision Transformer ViT-Base (in21k)
```powershell
python main_tf.py --model-name vit_base --epochs 15 --batch-size 32
```

### Model 4: Google Big Transfer BiT / ResNet152V2
```powershell
python main_tf.py --model-name bit_resnet152v2 --epochs 15 --batch-size 32
```

---

## 3. Running All Models & Comparing Results

```powershell
# Sequentially run all 4 models:
python main_tf.py --model-name all --epochs 15

# Generate 4-Model Comparison Leaderboard & Chart:
python tf_compare.py
```

All metrics summaries (`.json`), Confusion Matrices (`.png`), ROC Curves (`.png`), and Training Curves (`.png`) are saved in the `results/` folder.
