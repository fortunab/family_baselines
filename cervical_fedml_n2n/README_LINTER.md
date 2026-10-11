# 4-Stage Static Code Quality & Linter Architecture

To ensure code stability and maintainability, this project incorporates an automated 4-stage static code quality gate verified via `run_linter.py`.

---

## 🛡️ The 4 Verification Stages

```
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 1: Python AST Parsing                                 │
   │  - Parses every project .py file into an AST                │
   │  - Catches syntax errors, indentation issues, invalid tokens│
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 2: Ruff Linter Verification                           │
   │  - Modern high-performance linter checking PEP8 and bugs    │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 3: Flake8 Compliance Verification                     │
   │  - Strict PEP8 compliance across project modules            │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ Stage 4: Black Formatting Verification                      │
   │  - Enforces deterministic, unambiguous code formatting      │
   └─────────────────────────────────────────────────────────────┘
```

---

## 🏃 Running the Linter

```powershell
python run_linter.py
```
A returncode of `0` confirms zero warnings or errors across all 4 stages.
