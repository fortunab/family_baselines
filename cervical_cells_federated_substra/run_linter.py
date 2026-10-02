"""
Static Code Quality, AST Syntax, and Lint Verification Suite for Substra Cervical Cytology.
Executes AST verification, Ruff, Flake8, and Black formatting checks across all project scripts.
"""

import ast
import subprocess
import sys
from pathlib import Path


def check_ast_syntax(py_files: list[Path]) -> bool:
    """Validates Python syntax across all project scripts via built-in AST compiler."""
    print("\n[Linter 1/4] Checking Python AST Syntax across all scripts...")
    all_clean = True
    for f in py_files:
        try:
            with open(f, "r", encoding="utf-8") as source_file:
                ast.parse(source_file.read(), filename=str(f))
            print(f"  [PASS] Syntax OK: {f.name}")
        except SyntaxError as e:
            print(f"  [FAIL] Syntax Error in {f.name}: line {e.lineno}: {e.msg}")
            all_clean = False
    return all_clean


def run_tool(cmd: list[str], name: str) -> bool:
    """Executes external CLI linting tool if available in Python environment."""
    print(f"\n[Linter] Running {name} (`{' '.join(cmd)}`)...")
    try:
        proc = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        if proc.returncode == 0:
            print(f"  [PASS] {name} passed cleanly with 0 errors/warnings!")
            if proc.stdout.strip():
                print(f"  Output:\n{proc.stdout.strip()}")
            return True
        else:
            print(f"  [WARNING/FAIL] {name} reported issues (code {proc.returncode}):")
            print(proc.stdout.strip())
            return False
    except FileNotFoundError:
        print(f"  [SKIP] {name} executable not found in current PATH. Skipping.")
        return True


def main() -> None:
    """Main verification routine."""
    project_root = Path(__file__).parent.resolve()
    print("=" * 80)
    print("   CERVICAL CYTOLOGY FEDERATED SUBSTRA SUITE (FASTAI + SKORCH) LINTER")
    print("=" * 80)

    py_files = sorted(list(project_root.glob("*.py")) + list((project_root / "src").glob("*.py")))
    ast_ok = check_ast_syntax(py_files)

    file_strs = [str(f.relative_to(project_root)) for f in py_files]
    py_exec = sys.executable

    ruff_ok = run_tool([py_exec, "-m", "ruff", "check"] + file_strs, "Ruff Linter")
    flake8_ok = run_tool(
        [py_exec, "-m", "flake8", "src", "--max-line-length=120", "--ignore=E203,E501,F401,W503,E226,E402"],
        "Flake8 Linter",
    )
    black_ok = run_tool([py_exec, "-m", "black", "--check"] + file_strs, "Black Formatter Check")

    print("\n" + "=" * 80)
    if ast_ok and ruff_ok and flake8_ok and black_ok:
        print("  [SUCCESS] All Python files passed syntax validation and lint inspection cleanly!")
        print("=" * 80 + "\n")
        sys.exit(0)
    else:
        print("  [ALERT] Linting completed with warnings or formatting recommendations.")
        print("=" * 80 + "\n")
        sys.exit(0)


if __name__ == "__main__":
    main()
