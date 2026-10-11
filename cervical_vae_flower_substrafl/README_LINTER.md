# 4-Stage Static Quality Gate Architecture

Suite 27 implements a rigorous 4-stage static analysis pipeline to ensure strict production standards, complete syntax correctness, and uniform code styling.

---

## 🛡️ The 4 Quality Verification Stages

1. **Stage 1: AST (Abstract Syntax Tree) Verification**
   - Compiles all Python source files via `ast.parse()` to guarantee syntactical integrity and detect malformed blocks.
2. **Stage 2: Ruff Linter**
   - Ultra-fast Python linter checking syntax errors, unused imports, type annotation rules, and bad idioms with `max-line-length = 120`.
3. **Stage 3: Flake8 Linter**
   - Comprehensive style and PEP 8 auditor with line length allowance set to 120 and exclusion of build/environment folders (`.pixi`, `.venv`, `venv`).
4. **Stage 4: Black Code Formatter**
   - Enforces deterministic, consistent formatting using standard line length 120.

---

## 🏃 Running the Quality Gate

```powershell
# Run the 4-stage static quality checks:
python run_linter.py
```

Expected output:
```
================================================================================
           4-STAGE STATIC CODE QUALITY & LINTING VERIFICATION
================================================================================
[*] Target Directory: .../cervical_cells_vae_flower_substra

[Stage 1/4] Python Abstract Syntax Tree (AST) Validation...
  -> AST Verification PASSED (0 errors)

[Stage 2/4] Ruff Static Linter...
  -> Ruff Verification PASSED (0 errors)

[Stage 3/4] Flake8 Code Style Auditor...
  -> Flake8 Verification PASSED (0 errors)

[Stage 4/4] Black Code Formatting Check...
  -> Black Verification PASSED (0 errors)
================================================================================
[SUCCESS] ALL 4 STATIC QUALITY GATES PASSED (0 ERRORS DETECTED)
================================================================================
```
