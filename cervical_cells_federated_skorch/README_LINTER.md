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

### Auto-Formatting
```bash
black src main_federated_skorch.py compare_federated_strategies.py run_linter.py
ruff check --fix src main_federated_skorch.py compare_federated_strategies.py run_linter.py
```
