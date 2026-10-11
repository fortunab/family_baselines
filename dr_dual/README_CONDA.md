# Conda / Mamba Environment Guide

This document describes how to create and manage the isolated Conda environment using `environment.yml`.

---

## 1. Create Environment

```bash
conda env create -f environment.yml
```

Or with `mamba`:
```bash
mamba env create -f environment.yml
```

---

## 2. Activation

```bash
conda activate dr_dual_db_env
```

---

## 3. Verification

```bash
python -c "import numpy, sklearn, scipy, PIL; print('Conda setup verified successfully!')"
```
