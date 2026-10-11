# Code Quality & Linter Audit Guide: Alzheimer MRI Grayscale Benchmark

Instructions for running the 4-stage static quality audit.

---

## 1. Running the Quality Audit
```bash
python run_linter.py
```

Expected output:
```
======================================================================
ALZHEIMER MRI GRAYSCALE BENCHMARK - CODE QUALITY AUDIT
======================================================================
STAGE 1: Python AST Syntax & Compilation Verification
  [PASS] preprocess_grayscale_dataset.py
  [PASS] run_foundational_comparison.py
  [PASS] run_full_benchmark_suite.py
  [PASS] run_linter.py
  [PASS] run_ml_family_experiment.py
  [PASS] src\config.py
  [PASS] src\dataset.py
  [PASS] src\evaluator.py
  [PASS] src\feature_extractor.py
  [PASS] src\foundation_engine.py
  [PASS] src\ml_engine.py
  [PASS] src\preprocessing.py
  [PASS] src\visualizer.py
  [PASS] src\__init__.py
Stage 1 Result: 14/14 files passed (0 syntax errors).

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

## 2. Auto-Fixing Code Formatting
To format any future edits:
```bash
python -m ruff check --fix .
python -m black --line-length=120 .
```
