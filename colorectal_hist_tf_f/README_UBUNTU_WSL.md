# Running TensorFlow Colorectal Histology Baselines on Ubuntu / WSL (70/15/15 Split)

This guide walks you through setting up and running all 4 TensorFlow Foundation Models on the **Colorectal Histology dataset** inside **Ubuntu (WSL/WSL2)** using the **70% Train / 15% Val / 15% Test** evaluation protocol.

---

## 1. Access the Project in WSL

```bash
cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_tf_foundation
```

---

## 2. Setting Up Python & TensorFlow Environment

```bash
# 1. Update system packages
sudo apt update
sudo apt install -y python3 python3-pip python3-venv libgl1-mesa-glx

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Running the Foundation Models

```bash
# Model 1: Meta ConvNeXt-Large (70/15/15 Split)
python3 main_tf.py --model-name convnext_large --epochs 15 --batch-size 32

# Model 2: Google EfficientNetV2-L (70/15/15 Split)
python3 main_tf.py --model-name efficientnetv2_l --epochs 15 --batch-size 32

# Model 3: Google Vision Transformer ViT-Base (70/15/15 Split)
python3 main_tf.py --model-name vit_base --epochs 15 --batch-size 32

# Model 4: Google Big Transfer BiT / ResNet152V2 (70/15/15 Split)
python3 main_tf.py --model-name bit_resnet152v2 --epochs 15 --batch-size 32

# 4-Model Comparison Leaderboard & Chart
python3 tf_compare.py
```

---

## 4. Running in Background (`nohup` / `tmux`)

```bash
# Run training in background
nohup python3 main_tf.py --model-name convnext_large > convnext_tf.log 2>&1 &

# Monitor logs live
tail -f convnext_tf.log
```
