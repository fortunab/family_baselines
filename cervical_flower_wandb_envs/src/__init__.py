"""
Cervical Cytology Federated Learning with Flower (flwr) + fastai Package.
Pre-configures OpenMP runtimes and applies fastcore read-only docstring patch.
"""

import os

# Prevent OpenMP multiple runtime conflict on Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Apply fastcore docstring patch for Python 3.12+ and PyTorch 2.4+ compatibility
try:
    import fastcore.foundation as _fcf

    if hasattr(_fcf, "add_docs"):
        _orig_add_docs = _fcf.add_docs

        def _safe_add_docs(cls, **docs):
            try:
                _orig_add_docs(cls, **docs)
            except (AttributeError, TypeError):
                pass

        _fcf.add_docs = _safe_add_docs
except Exception:
    pass
