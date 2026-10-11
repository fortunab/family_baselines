# Windows 11 Native Execution Guide

This guide covers running Suite 27 (`cervical_cells_vae_flower_substra`) natively on Windows 11 using PowerShell.

---

## 🛠️ Windows-Specific Optimizations

1. **OpenMP Deduplication**:
   - `src/__init__.py` and all entrypoints automatically set `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` to prevent multiple OpenMP runtime collisions (common when combining NumPy, PyTorch, and fastai on Windows).

2. **File Path Handling**:
   - All modules use `pathlib.Path` or `os.path.join` to cleanly handle Windows backslashes and drive letters.

3. **PowerShell Script Execution Policy**:
   - If running setup scripts fails due to execution policy:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
   .\scripts\setup_pixi.ps1
   # or
   .\scripts\setup_conda.ps1
   # or
   .\scripts\setup_venv.ps1
   ```

4. **CUDA & CPU Fallback**:
   - The framework automatically detects CUDA availability. If an NVIDIA GPU is not detected, it gracefully falls back to CPU execution without manual flag toggles.
