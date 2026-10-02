# Static Code Quality, Linting & Formatting Standards Guide

This guide describes the 4-stage static quality verification pipeline implemented in `run_linter.py`.

---

## 1. Quality Objectives

Medical AI systems demand the highest standard of code integrity and deterministic behavior:
- Zero syntax errors or deprecated AST structures.
- Strict import hygiene and clean namespace management.
- Formatting compliance adhering to PEP 8 and the Black specification.
- Zero linter errors across all codebase components.

---

## 2. The 4 Verification Stages

### Stage 1: Python AST Syntax Validation
Compiles every Python source file into an Abstract Syntax Tree via `ast.parse()`. Validates lexical correctness across all modules.

### Stage 2: Ruff Linter (`ruff check .`)
Enforces modern Python standards (`E`, `F`, `W`, `I` rules) at high speed:
- `E` / `W`: PEP 8 styling rules.
- `F`: Pyflakes error checks (undefined names, unused variables).
- `I`: isort automated import ordering.

### Stage 3: Flake8 Compliance (`flake8 .`)
Audits compatibility across legacy and modern Python conventions.
- Configured ignores: `E203,E501,F401,W503,E226,E402`.

### Stage 4: Black Formatter (`black --check .`)
Validates that code formatting matches the uncompromising Black code style with 120-character line lengths.

---

## 3. Running the Pipeline

To run the complete 4-stage verification suite:
```powershell
python run_linter.py
```

Expected output:
```
================================================================================
  ALL 4 CODE QUALITY & LINTER STAGES PASSED CLEANLY (0 ERRORS)
================================================================================
```
