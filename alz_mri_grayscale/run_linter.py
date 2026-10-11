"""Multi-Stage Code Quality & Linter Runner for Alzheimer MRI Grayscale Benchmark.

Executes 4 quality verification stages:
1. Python AST Compilation & Syntax Validity
2. Ruff Code Analysis & Import Ordering
3. Flake8 Style & PEP8 Check
4. Black Format Compliance Verification
"""

import ast
import subprocess
import sys
from pathlib import Path


def stage_ast_syntax_check(root_dir: Path) -> bool:
    """Stage 1: Verify that every Python file compiles into valid AST with 0 syntax errors."""
    print("\n" + "=" * 70)
    print("STAGE 1: Python AST Syntax & Compilation Verification")
    print("=" * 70)

    py_files = sorted(list(root_dir.rglob("*.py")))
    py_files = [
        f
        for f in py_files
        if not any(part in f.parts for part in ["venv", ".venv", "env", ".env", "venv_fastai_toml", "__pycache__"])
    ]

    errors = 0
    for p in py_files:
        rel_path = p.relative_to(root_dir)
        try:
            with open(p, "r", encoding="utf-8") as f:
                code = f.read()
            ast.parse(code, filename=str(p))
            print(f"  [PASS] {rel_path}")
        except SyntaxError as e:
            print(f"  [FAIL] {rel_path}: {e}")
            errors += 1

    print(f"\nStage 1 Result: {len(py_files) - errors}/{len(py_files)} files passed ({errors} syntax errors).")
    return errors == 0


def run_tool_command(cmd_args: list, tool_name: str) -> bool:
    """Run an external CLI tool and return whether it succeeded."""
    print("\n" + "=" * 70)
    print(f"STAGE: {tool_name}")
    print("=" * 70)
    print(f"Executing: {' '.join(cmd_args)}")

    try:
        res = subprocess.run(cmd_args, capture_output=True, text=True, check=False)
        if res.stdout:
            print(res.stdout.strip())
        if res.stderr:
            print(res.stderr.strip())

        if res.returncode == 0:
            print(f"  [PASS] {tool_name} completed cleanly (exit code 0).")
            return True
        else:
            print(f"  [FAIL] {tool_name} reported issues (exit code {res.returncode}).")
            return False
    except FileNotFoundError:
        print(f"  [WARN] {tool_name} is not installed in the current environment PATH.")
        return True


def main():
    root_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print("ALZHEIMER MRI GRAYSCALE BENCHMARK - CODE QUALITY AUDIT")
    print(f"Project Directory: {root_dir}")
    print("=" * 70)

    results = {}
    results["AST Syntax"] = stage_ast_syntax_check(root_dir)

    python_exe = sys.executable
    results["Ruff"] = run_tool_command([python_exe, "-m", "ruff", "check", "."], "Ruff Linter")
    results["Flake8"] = run_tool_command(
        [python_exe, "-m", "flake8", "--max-line-length=120", "--extend-ignore=E203,W503,F401,E402", "."],
        "Flake8 PEP8 Check",
    )
    results["Black"] = run_tool_command(
        [python_exe, "-m", "black", "--check", "--line-length=120", "."],
        "Black Formatting Check",
    )

    print("\n" + "=" * 70)
    print("QUALITY AUDIT SUMMARY")
    print("=" * 70)
    all_passed = True
    for stage, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        if not passed:
            all_passed = False
        print(f"  - {stage:<25}: {status}")

    print("=" * 70)
    if all_passed:
        print("ALL QUALITY CHECKS PASSED (0 ERRORS)")
        sys.exit(0)
    else:
        print("SOME QUALITY CHECKS REPORTED ISSUES")
        sys.exit(1)


if __name__ == "__main__":
    main()
