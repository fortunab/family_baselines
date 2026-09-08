"""
Herlev Cervical Cytology Pathology Foundation Suite with fastai, TOML & Weights & Biases.
"""

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Compatibility patch for fastcore & PyTorch / Python 3.12+ (read-only __doc__ on C-extension functions)
try:
    import fastcore.foundation

    def _safe_add_docs(cls, cls_doc=None, **docs):
        if cls_doc is not None:
            try:
                cls.__doc__ = cls_doc
            except (AttributeError, TypeError):
                pass
        for k, v in docs.items():
            try:
                f = getattr(cls, k)
                f.__doc__ = v
            except (AttributeError, TypeError):
                pass

    fastcore.foundation.add_docs = _safe_add_docs
except Exception:
    pass
