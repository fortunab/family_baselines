# Pixi Environment Guide

Pixi is a fast conda-ecosystem package management tool providing cross-platform dependency locking for `win-64` and `linux-64`.

---

## 🚀 Installation & Setup

### Windows PowerShell
```powershell
.\scripts\setup_pixi.ps1
```

### Linux / WSL / macOS
```bash
bash scripts/setup_pixi.sh
```

---

## ⚡ Configured Pixi Tasks

```powershell
# Run baseline raw micrographs
pixi run test-raw

# Run with Noise2Noise denoising
pixi run test-n2n

# Run with CycleGAN stain harmonization
pixi run test-cyclegan

# Run with dual integrated pipeline
pixi run test-dual

# Run skorch backend with dual preprocessing
pixi run test-skorch

# Run comparative benchmark across all 4 modes
pixi run compare

# Run 4-stage static code quality & linter check
pixi run lint
```
