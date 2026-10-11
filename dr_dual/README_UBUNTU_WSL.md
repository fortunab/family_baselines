# Linux / Ubuntu / WSL2 Execution Guide

This guide details running the Dual-Database Clinical Machine Learning Framework on Linux, Ubuntu 22.04 LTS, or WSL2.

---

## 1. System Dependencies

```bash
sudo apt-get update && sudo apt-get install -y \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    build-essential
```

---

## 2. Environment Initialization

```bash
chmod +x scripts/*.sh
bash scripts/setup_venv.sh
source venv/bin/activate
```

---

## 3. Benchmark Execution

```bash
# 1. Synthesize Dual Databases
python3 generate_dual_databases.py

# 2. Run Diabetic Retinopathy ML Family Benchmark
python3 run_ml_family_benchmark.py

# 3. Run Dermatology Dual-Database Benchmark
python3 run_isic_ham_benchmark.py

# 4. Verify Static Code Quality
python3 run_linter.py
```
