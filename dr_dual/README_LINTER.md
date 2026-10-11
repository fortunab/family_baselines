# Static Code Quality & Linter Audit Guide

This repository maintains strict production-grade code quality verified across four static analysis stages.

---

## 1. Automated Linter Runner

Execute the static linter script:
```bash
python run_linter.py
```

---

## 2. Verification Stages

1. **AST Syntactic Validation**: Parses the Abstract Syntax Tree of every `.py` file to guarantee valid syntax and prevent compilation regressions.
2. **Ruff**: High-performance Rust-based Python linter enforcing PEP 8 standards, import cleanliness, and code smell elimination.
3. **Flake8**: Validates complexity, style guides, and potential variable leakage.
4. **Black Check**: Verifies PEP 8 formatting and 120-character line length compliance.

---

## 3. Manual Lint Commands

```bash
# Run Ruff
ruff check .

# Run Flake8
flake8 . --max-line-length=120 --ignore=E501,W503,F401

# Run Black Formatting Check
black --check . --line-length=120
```
