# Code Quality and Lint Verification Guide: Substra Cervical Cytology

This suite includes a unified 4-stage static code quality auditor:

```
[run_linter.py]
  ├── Stage 1: Python AST Syntax Checker (all 14 scripts)
  ├── Stage 2: Ruff Linter (PEP compliance & imports)
  ├── Stage 3: Flake8 Linter (formatting & syntax)
  └── Stage 4: Black Formatter Validation
```

---

## 1. Running the Verification Suite

```powershell
python run_linter.py
```

Expected Clean Output:
```
================================================================================
   CERVICAL CYTOLOGY FEDERATED SUBSTRA SUITE (FASTAI + SKORCH) LINTER
================================================================================

[Linter 1/4] Checking Python AST Syntax across all scripts...
  [PASS] Syntax OK: compare_substra_strategies.py
  [PASS] Syntax OK: main_federated_substra.py
  [PASS] Syntax OK: run_linter.py
  [PASS] Syntax OK: __init__.py
  [PASS] Syntax OK: dataset.py
  [PASS] Syntax OK: evaluator.py
  [PASS] Syntax OK: fastai_engine.py
  [PASS] Syntax OK: seed_selector.py
  [PASS] Syntax OK: skorch_engine.py
  [PASS] Syntax OK: substra_algo.py
  [PASS] Syntax OK: substra_opener.py
  [PASS] Syntax OK: substra_orchestrator.py
  [PASS] Syntax OK: toml_config.py
  [PASS] Syntax OK: wandb_tracker.py

[Linter] Running Ruff Linter...
  [PASS] Ruff Linter passed cleanly with 0 errors/warnings!

[Linter] Running Flake8 Linter...
  [PASS] Flake8 Linter passed cleanly with 0 errors/warnings!

[Linter] Running Black Formatter Check...
  [PASS] Black Formatter Check passed cleanly with 0 errors/warnings!

================================================================================
  [SUCCESS] All Python files passed syntax validation and lint inspection cleanly!
================================================================================
```
