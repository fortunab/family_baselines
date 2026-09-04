# Code Quality, Formatting & Linter Guide for skorch Suite

This repository adheres to strict PEP 8, typing, and style conventions using modern Python tooling configured via `pyproject.toml`.

---

## 🛠️ Included Quality Tools

1. **Ruff**: Ultra-fast linter replacing Flake8, isort, and pyflakes.
2. **Flake8**: Standard PEP 8 static analysis checker.
3. **Black**: Deterministic code formatter enforcing 100-character line lengths.
4. **Isort**: Automatic import organization.
5. **MyPy**: Static type checking for type hints.

---

## 🚀 How to Run the Linter

```powershell
python run_linter.py
```

### Manual Individual Commands:
```bash
# Ruff linting:
ruff check .

# Black format check:
black --check .

# Flake8 check:
flake8 src --max-line-length=120 --ignore=E501,F401,W503,E226
```
