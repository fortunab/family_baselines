"""
Cervical Cytology Federated Learning FedML Suite.
Initializes the package with OpenMP duplicate library protection for Windows environments.
"""

import os

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

__version__ = "1.0.0"
