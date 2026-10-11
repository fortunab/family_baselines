"""
Cervical Cytology Federated Flower & Substra Learning with VAE Preprocessing Package.
Ensures OpenMP runtime deduplication on Windows/Linux environments.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "1.0.0"
