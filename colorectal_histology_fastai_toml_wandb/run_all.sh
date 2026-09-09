# 1. Virtual Environment Setup
python3 -m venv venv_fastai_toml
source venv_fastai_toml/bin/activate

# 2. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. Run Code Quality Linter
python3 run_linter.py

# 4. Run Training
python3 main_fastai_wandb.py --config configs/convnext_base.toml
python3 main_fastai_wandb.py --config configs/vit_base.toml
python3 compare_fastai_models.py
