# Code Quality & Linter Audit Guide: Alzheimer MRI fastai Suite

This project adheres to rigorous code quality gates verified by `run_linter.py` across 4 progressive stages.

---

## 1. Quick Quality Check
To execute all 4 stages in a single pass:
```bash
python run_linter.py
```

Expected output:
```
======================================================================
ALZHEIMER MRI FASTAI CAM SUITE - CODE QUALITY AUDIT
======================================================================
STAGE 1: Python AST Syntax & Compilation Verification
  [PASS] compare_cam_metrics.py
  [PASS] run_linter.py
  [PASS] train_and_evaluate.py
  [PASS] visualize_cam_heatmaps.py
  ...
Stage 1 Result: 15/15 files passed (0 syntax errors).

STAGE: Ruff Linter
  [PASS] Ruff Linter completed cleanly (exit code 0).

STAGE: Flake8 PEP8 Check
  [PASS] Flake8 PEP8 Check completed cleanly (exit code 0).

STAGE: Black Formatting Check
  [PASS] Black Formatting Check completed cleanly (exit code 0).

======================================================================
QUALITY AUDIT SUMMARY
======================================================================
  - AST Syntax               : PASSED
  - Ruff                     : PASSED
  - Flake8                   : PASSED
  - Black                    : PASSED
======================================================================
ALL QUALITY CHECKS PASSED (0 ERRORS)
```

---

## 2. Configuration Specifications

### Ruff (`pyproject.toml`)
- Line length: 120
- Target version: Python 3.11
- Enforces strict import ordering (isort compatibility) and pycodestyle warnings.

### Flake8
- Line length: 120
- Extended ignore list: `E203, W503, F401` (harmonized with Black formatting).

### Black
- Deterministic code formatting with `--line-length=120`.

---

## 3. Auto-Fixing Code Formatting
If modifying source code, automatically format and clean imports using:
```bash
python -m ruff check --fix .
python -m black --line-length=120 .
```
