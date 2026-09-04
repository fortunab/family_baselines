# Automated Code Quality & Linter Guide

The suite enforces strict code quality, PEP 8 compliance, and type checking using industry-standard linters configured in `pyproject.toml`.

---

## 1. Multi-Linter Tooling

- **AST Syntax Check**: Native Python `ast.parse` validates all Python files for syntax errors.
- **Ruff**: Ultra-fast Rust-based Python linter checking syntax, bugs, imports (`I`), and flake8 conventions (`E`, `F`, `W`, `B`).
- **Flake8**: Standard PEP 8 style guide enforcement with custom line length rules.
- **Black**: The uncompromising Python code formatter verifying formatting consistency.
- **MyPy**: Static type analysis.

---

## 2. Running the Linter

Execute the one-click runner:

```powershell
python run_linter.py
```

Expected clean output:
```
================================================================================
     HERLEV CYTOLOGY PATHOLOGY FOUNDATION FASTAI LINTER SUITE
================================================================================

[Linter 1/4] Checking Python AST Syntax across all scripts...
  [PASS] Syntax OK: compare_herlev_models.py
  [PASS] Syntax OK: main_herlev_fastai.py
  [PASS] Syntax OK: run_linter.py
  ...
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

---

## 3. Auto-Formatting Code

To auto-format code according to the repository rules:

```powershell
black src main_herlev_fastai.py compare_herlev_models.py run_linter.py
ruff check --fix src main_herlev_fastai.py compare_herlev_models.py run_linter.py
```
