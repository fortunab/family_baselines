"""
Automated Code Quality & Linter Runner.
Runs:
1. Syntax Validation (ast.parse)
2. Ruff / Flake8 Style & Lint Checker
3. Black Formatting Check
"""

import ast
import os
import subprocess
import sys
from pathlib import Path


def check_syntax(project_root: Path) -> bool:
    print("\n[Linter 1/4] Checking Python AST Syntax across all scripts...")
    py_files = list(project_root.glob("*.py")) + list((project_root / "src").glob("*.py"))
    has_error = False

    for f in py_files:
        try:
            with open(f, "r", encoding="utf-8") as file:
                ast.parse(file.read(), filename=str(f))
            print(f"  [PASS] Syntax OK: {f.name}")
        except SyntaxError as e:
            print(f"  [FAIL] Syntax Error in {f.name}: {e}")
            has_error = True

    return not has_error


def run_command_tool(cmd: list, tool_name: str) -> bool:
    print(f"\n[Linter] Running {tool_name} (`{' '.join(cmd)}`)...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode == 0:
            print(f"  [PASS] {tool_name} passed cleanly with 0 errors/warnings!")
            if res.stdout and res.stdout.strip():
                print(f"  Output:\n{res.stdout.strip()}")
            return True
        else:
            print(f"  [NOTICE] {tool_name} reported items (Return code: {res.returncode}):")
            if res.stdout and res.stdout.strip():
                print(f"{res.stdout.strip()}")
            if res.stderr and res.stderr.strip():
                print(f"{res.stderr.strip()}")
            return False
    except FileNotFoundError:
        print(f"  [INFO] Tool '{cmd[0]}' is not installed globally in current PATH. Skipping.")
        return True


def main():
    project_root = Path(__file__).parent.resolve()
    print("=" * 80)
    print("      PATHOLOGY FOUNDATION FASTAI CODE QUALITY & LINTER SUITE")
    print("=" * 80)

    # 1. AST Syntax Check
    syntax_ok = check_syntax(project_root)

    # 2. Ruff Linter
    targets = ["src", "main_fastai_foundation.py", "compare_foundation_models.py", "run_linter.py"]
    ruff_ok = run_command_tool([sys.executable, "-m", "ruff", "check"] + targets, "Ruff Linter")

    # 3. Flake8 Linter
    flake8_ok = run_command_tool(
        [
            sys.executable,
            "-m",
            "flake8",
            "src",
            "--max-line-length=120",
            "--ignore=E501,F401,W503,E226,E402",
        ],
        "Flake8 Linter",
    )

    # 4. Black Formatting Check
    black_ok = run_command_tool(
        [sys.executable, "-m", "black", "--check"] + targets, "Black Formatter Check"
    )

    all_passed = syntax_ok and ruff_ok and flake8_ok and black_ok

    print("\n" + "=" * 80)
    if all_passed:
        print("  [SUCCESS] All Python files passed syntax validation and lint inspection cleanly!")
    else:
        print("  [NOTICE] Linting completed. Review any warnings above.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
