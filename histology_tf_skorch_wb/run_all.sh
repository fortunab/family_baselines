cd /mnt/c/Users/Lenovo/.gemini/antigravity/scratch/colorectal_histology_skorch_toml_wandb

# 1. Virtual Environment Setup
python3 -m venv venv_skorch_toml
source venv_skorch_toml/bin/activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt


# 3. Run Training
python3 main_skorch_wandb.py --config configs/convnext_base.toml
python3 main_skorch_wandb.py --config configs/resnet50d.toml
python3 compare_skorch_models.py
