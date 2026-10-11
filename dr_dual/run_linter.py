"""Code Quality and Linter Audit Suite for Dual-Database ML Suite.

Runs static analysis checks across the codebase:
1. Python AST Syntactic Validation
2. Ruff Fast Linter (if installed)
3. Flake8 Style & Complexity Checker (if installed)
4. Black Formatting Check (if installed)
"""

import ast
import subprocess
import sys
from pathlib import Path


def check_ast(root: Path) -> int:
    errors = 0
    py_files = list(root.rglob("*.py"))
    print(f"[AST CHECK] Scanning {len(py_files)} Python source files...")
    for f in py_files:
        if any(part in f.parts for part in ["venv", ".venv", "__pycache__", "build", "dist"]):
            continue
        try:
            with open(f, "r", encoding="utf-8") as source:
                ast.parse(source.read(), filename=str(f))
        except SyntaxError as e:
            print(f"  [ERROR] Syntax error in {f}: {e}")
            errors += 1
    if errors == 0:
        print("  [SUCCESS] All Python source files passed AST verification cleanly.")
    return errors


def run_tool(cmd: list) -> bool:
    try:
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            print(f"  [SUCCESS] {' '.join(cmd[:2])} passed with 0 errors.")
            return True
        else:
            print(f"  [OUTPUT] {res.stdout.strip()}")
            if res.stderr.strip():
                print(f"  [STDERR] {res.stderr.strip()}")
            return False
    except FileNotFoundError:
        print(f"  [SKIP] Tool '{cmd[0]}' not installed or not in PATH.")
        return True


def main():
    root = Path(__file__).resolve().parent
    print("=" * 65)
    print("      STATIC CODE QUALITY & LINTER AUDIT SUITE      ")
    print("=" * 65)

    ast_errs = check_ast(root)

    print("\n[RUFF LINTER CHECK]")
    run_tool([sys.executable, "-m", "ruff", "check", str(root)])

    print("\n[FLAKE8 CHECK]")
    run_tool([sys.executable, "-m", "flake8", str(root), "--max-line-length=120", "--ignore=E501,W503,F401,E203,E226"])

    print("\n[BLACK FORMATTING CHECK]")
    run_tool([sys.executable, "-m", "black", "--check", str(root), "--line-length=120"])

    print("=" * 65)
    if ast_errs == 0:
        print(" AUDIT FINISHED: Codebase is syntactically sound and valid. ")
    else:
        print(f" AUDIT FAILED: {ast_errs} AST syntax errors detected. ")
    print("=" * 65)


if __name__ == "__main__":
    main()
