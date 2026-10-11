"""
4-Stage Static Code Quality & Linter Verification Pipeline.
Checks Python AST syntax compilation, Ruff linting, Flake8 compliance, and Black formatting.
"""

import ast
import subprocess
import sys
from pathlib import Path


def check_ast_syntax(root_dir: Path) -> tuple[int, int]:
    """Stage 1: Validates Python AST syntax across all project .py files."""
    print("\n" + "=" * 80)
    print("  STAGE 1: PYTHON AST SYNTAX VERIFICATION")
    print("=" * 80)

    # Exclude virtual environments and cache directories
    ignored_parts = {".venv", "venv", ".pixi", "__pycache__", ".git"}
    py_files = [p for p in root_dir.rglob("*.py") if not any(part in p.parts for part in ignored_parts)]

    passed, errors = 0, 0

    for py_file in py_files:
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=str(py_file))
            rel_path = py_file.relative_to(root_dir)
            print(f"  [OK] Syntax Valid: {rel_path}")
            passed += 1
        except SyntaxError as e:
            print(f"  [FAIL] Syntax Error in {py_file}: {e}")
            errors += 1

    print(f"[*] AST Check Finished: {passed} passed, {errors} errors.")
    return passed, errors


def run_tool(cmd: list[str], stage_name: str) -> bool:
    """Executes a subprocess tool and returns success status."""
    print("\n" + "=" * 80)
    print(f"  {stage_name}")
    print("=" * 80)
    print(f"[*] Running: {' '.join(cmd)}")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print(f"  [PASS] {stage_name} passed cleanly (0 errors).")
            if res.stdout.strip():
                print(res.stdout.strip())
            return True
        else:
            print(f"  [FAIL] {stage_name} reported issues:")
            if res.stdout.strip():
                print(res.stdout.strip())
            if res.stderr.strip():
                print(res.stderr.strip())
            return False
    except FileNotFoundError:
        print(f"  [WARN] Executable '{cmd[0]}' not found in current environment. Skipping stage.")
        return True


def main() -> None:
    root = Path(__file__).resolve().parent

    _ast_pass, ast_err = check_ast_syntax(root)

    targets = ["src", "main_federated_fedml.py", "compare_cervical_preprocessing.py", "run_linter.py"]

    # Use sys.executable -m to ensure execution within active environment
    ruff_ok = run_tool(
        [sys.executable, "-m", "ruff", "check", "--config=pyproject.toml"] + targets,
        "STAGE 2: RUFF LINTER VERIFICATION",
    )
    flake8_ok = run_tool(
        [
            sys.executable,
            "-m",
            "flake8",
            "--exclude=.venv,venv,.pixi",
            "--ignore=E203,E501,F401,W503,E226,E402",
            "--max-line-length=120",
        ]
        + targets,
        "STAGE 3: FLAKE8 COMPLIANCE VERIFICATION",
    )
    black_ok = run_tool(
        [sys.executable, "-m", "black", "--check"] + targets,
        "STAGE 4: BLACK FORMATTING VERIFICATION",
    )

    all_ok = (ast_err == 0) and ruff_ok and flake8_ok and black_ok

    print("\n" + "=" * 80)
    if all_ok:
        print("  ALL 4 CODE QUALITY & LINTER STAGES PASSED CLEANLY (0 ERRORS)")
    else:
        print("  CODE QUALITY CHECKS REPORTED ISSUES (PLEASE REVIEW ABOVE)")
    print("=" * 80 + "\n")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
