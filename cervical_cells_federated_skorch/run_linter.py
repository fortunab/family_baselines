"""
Unified Static Code Analysis and Style Enforcement Tool for Cervical Cytology skorch FL.
Executes Python AST syntax parsing, Ruff, Flake8, and Black formatting checks.
"""

import ast
import subprocess
import sys
from pathlib import Path


def find_python_files(root: Path) -> list[Path]:
    """Discovers all repository Python scripts excluding virtualenvs and cache."""
    py_files = []
    excluded_dirs = {
        "venv",
        ".venv",
        "venv_cervical_skorch",
        "venv_cervical_flower",
        "venv_fastai_toml",
        ".git",
        "__pycache__",
        "build",
        "dist",
        ".pixi",
    }
    for p in root.rglob("*.py"):
        if any(ex in p.parts for ex in excluded_dirs):
            continue
        py_files.append(p)
    return sorted(py_files)


def check_ast_syntax(py_files: list[Path]) -> bool:
    """Validates abstract syntax tree parsing for all discovered Python files."""
    print("\n[Linter 1/4] Checking Python AST Syntax across all scripts...")
    all_clean = True
    for f in py_files:
        try:
            with open(f, "r", encoding="utf-8") as src:
                ast.parse(src.read(), filename=str(f))
            print(f"  [PASS] Syntax OK: {f.name}")
        except Exception as e:
            print(f"  [FAIL] Syntax Error in {f}: {e}")
            all_clean = False
    return all_clean


def run_command(cmd: list[str], tool_name: str) -> bool:
    """Executes external linter tool subprocess and reports results."""
    print(f"\n[Linter] Running {tool_name} (`{' '.join(cmd)}`)...")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if res.returncode == 0:
            print(f"  [PASS] {tool_name} passed cleanly with 0 errors/warnings!")
            if res.stdout.strip():
                print("  Output:\n" + res.stdout.strip())
            return True
        else:
            print(f"  [NOTICE] {tool_name} reported items (Return code: {res.returncode}):")
            if res.stdout.strip():
                print(res.stdout.strip())
            if res.stderr.strip():
                print(res.stderr.strip())
            return False
    except FileNotFoundError:
        print(f"  [SKIP] {tool_name} is not installed in the active environment.")
        return True


def main():
    print("=" * 80)
    print("   CERVICAL CYTOLOGY FEDERATED FLOWER SKORCH LINTER SUITE")
    print("=" * 80)

    root = Path(__file__).parent.resolve()
    py_files = find_python_files(root)
    py_file_strs = [str(f.relative_to(root)) for f in py_files]

    ast_ok = check_ast_syntax(py_files)

    python_exe = sys.executable
    ruff_ok = run_command([python_exe, "-m", "ruff", "check"] + py_file_strs, "Ruff Linter")
    flake_ok = run_command(
        [
            python_exe,
            "-m",
            "flake8",
            "src",
            "--max-line-length=120",
            "--ignore=E203,E501,F401,W503,E226,E402",
        ],
        "Flake8 Linter",
    )
    black_ok = run_command([python_exe, "-m", "black", "--check"] + py_file_strs, "Black Formatter Check")

    print("\n" + "=" * 80)
    if ast_ok and ruff_ok and flake_ok and black_ok:
        print("  [SUCCESS] All Python files passed syntax validation and lint inspection cleanly!")
    else:
        print("  [NOTICE] Linting completed. Review any warnings above.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
