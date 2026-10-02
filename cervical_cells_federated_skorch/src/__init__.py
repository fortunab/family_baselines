"""
Cervical Cytology Federated Learning Suite (Flower + skorch).
Ensures OpenMP and multiprocessing compatibility on Windows and POSIX systems.
"""

import os

# Prevent Intel OpenMP duplicate runtime initialization conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
