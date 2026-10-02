# Code Quality & Linter Architecture Guide

This suite incorporates a unified static code analysis tool (`run_linter.py`) to enforce code quality, clean PEP 8 formatting, and type-hint consistency across Python scripts.

---

## 1. Quality Tools Evaluated

The linter runner executes 4 auditing stages:
1. **Python AST Syntax Check**: Parses every `.py` file to verify grammatical correctness and abstract syntax tree integrity.
2. **Ruff**: Extremely fast Rust-based linter checking for undefined variables, unused imports, exception anti-patterns, and modern Python idioms.
3. **Flake8**: Traditional PEP 8 style guide enforcement and cyclomatic complexity auditing.
4. **Black**: Uncompromising code formatter check verifying consistent indentation, bracket placement, and line length rules.

---

## 2. Running the Linter

```bash
python run_linter.py
```

### Expected Clean Output:
```
================================================================================
          CERVICAL FLOWER FEDERATED SUITE - CODE QUALITY AUDITOR
================================================================================
Found 11 Python files to inspect.

[STAGE 1/4] Python Abstract Syntax Tree (AST) Validation...
  PASS: 11/11 files syntactically valid.

[STAGE 2/4] Ruff Linter Inspection...
  PASS: Ruff reported 0 violations.

[STAGE 3/4] Flake8 Code Style Audit...
  PASS: Flake8 reported 0 violations.

[STAGE 4/4] Black Formatting Verification...
  PASS: Black verified all files are properly formatted.

================================================================================
  AUDIT RESULT: ALL CODE QUALITY CHECKS PASSED (0 ERRORS)
================================================================================
```

---

## 3. Auto-Formatting

If edits are introduced that violate formatting standards:
```bash
# Auto-format with Black:
black .

# Auto-fix Ruff violations:
ruff check --fix .
```
