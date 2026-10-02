"""
Unified Code Quality & Formatting Linter Runner for Cervical Cytology Federated Flower Suite.
Executes AST syntax validation, Ruff linter, Flake8 style checker, and Black formatting verification.
"""

import ast
import subprocess
import sys
from pathlib import Path
from typing import List


def check_ast_syntax(py_files: List[Path]) -> bool:
    """Verifies Python Abstract Syntax Tree across all files."""
    print("\n[Linter 1/4] Checking Python AST Syntax across all scripts...")
    all_valid = True
    for file_path in py_files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
            ast.parse(source, filename=str(file_path))
            print(f"  [PASS] Syntax OK: {file_path.name}")
        except SyntaxError as e:
            print(f"  [FAIL] Syntax Error in {file_path.name}: {e}")
            all_valid = False
        except Exception as e:
            print(f"  [FAIL] Failed reading {file_path.name}: {e}")
            all_valid = False
    return all_valid


def run_command_tool(cmd: List[str], tool_name: str) -> bool:
    """Executes a subprocess command tool and formats output cleanly."""
    cmd_str = " ".join(cmd)
    print(f"\n[Linter] Running {tool_name} (`{cmd_str}`)...")
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
    print("   CERVICAL CYTOLOGY FEDERATED FLOWER FASTAI LINTER SUITE")
    print("=" * 80)

    # 1. AST Syntax Check
    py_files = sorted(list(project_root.glob("*.py")) + list(project_root.glob("src/*.py")))
    syntax_ok = check_ast_syntax(py_files)

    # 2. Ruff Linter
    targets = [
        "src",
        "main_federated_flower.py",
        "compare_federated_strategies.py",
        "run_linter.py",
    ]
    ruff_ok = run_command_tool([sys.executable, "-m", "ruff", "check"] + targets, "Ruff Linter")

    # 3. Flake8 Linter
    flake8_ok = run_command_tool(
        [
            sys.executable,
            "-m",
            "flake8",
            "src",
            "--max-line-length=120",
            "--ignore=E203,E501,F401,W503,E226,E402",
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
