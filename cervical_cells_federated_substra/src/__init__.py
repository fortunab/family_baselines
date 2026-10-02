"""
Cervical Cytology Federated Learning Substra Suite (fastai + skorch).
Multi-environment: Pixi, uv, Conda, venv, Pyenv.
Dynamic high-entropy seeds with Random Best Select (No Seed 42).
"""

import os

# Prevent OpenMP duplicate library crashes on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
