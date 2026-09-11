# Activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Tier 1: Classical SVM
python3 main_svm.py

# Tier 2: ConvNeXt-Tiny
python3 main_convnext.py --model-name convnext_tiny --epochs 15 --batch-size 32

# Tier 3: Vision Transformer
python3 main_vit.py --model-name vit_base_patch16_224 --epochs 15 --batch-size 32

# Tier 4: Foundation Model + Linear Probe
python3 main_foundation.py --model-name dinov2_base

# Compare all 4 baselines side-by-side
python3 compare_baselines.py
