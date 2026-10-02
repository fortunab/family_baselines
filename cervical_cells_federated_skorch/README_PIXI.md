# Pixi Environment & Workflow Guide for skorch Federated Learning

[Pixi](https://pixi.sh/) is a high-performance, cross-platform package management and task running tool built on the Conda ecosystem with deterministic lockfiles.

---

## 1. Quickstart (Automated Scripts)

### Windows PowerShell
```powershell
.\setup_pixi.ps1
```
*`setup_pixi.ps1` is fully self-bootstrapping: it automatically verifies if Pixi is installed, downloads it if absent, registers it into your active session PATH, and runs `pixi install`.*

### Linux / WSL2 / macOS Bash
```bash
chmod +x setup_pixi.sh
./setup_pixi.sh
```

---

## 2. Enabling Pixi Manually in Existing PowerShell Session

If you just installed Pixi, refresh your current terminal PATH:
```powershell
$env:Path = "$HOME\.pixi\bin;$env:Path"
pixi --version
```

---

## 3. Pixi Tasks & Workflow

`pixi.toml` contains pre-configured tasks for simulation, training, benchmarking, and linting:

```bash
# 1. Fast Dry-Run Verification (2 rounds, 3 clients)
pixi run dry-run

# 2. Full Federated Training (FedAvg with ConvNeXt)
pixi run train

# 3. Strategy Benchmark (FedAvg vs FedProx vs FedAdam)
pixi run benchmark

# 4. Code Quality Audit (AST, Ruff, Flake8, Black)
pixi run lint

# 5. Interactive Shell
pixi shell
```

---

## 4. Multi-Platform Lockfile (`pixi.lock`)

Pixi creates a reproducible, cross-platform `pixi.lock` file locking exact binary packages across `win-64` and `linux-64`.
To update the lockfile after altering `pixi.toml`:
```bash
pixi lock
```
