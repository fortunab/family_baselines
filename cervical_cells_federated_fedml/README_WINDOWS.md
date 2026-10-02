# Windows Native Environment & GPU Execution Guide

This guide covers optimal configuration, OpenMP safeguards, and execution on native Windows 11 / Windows 10 environments.

---

## 1. OpenMP Runtime Conflict Prevention

On Windows systems with Anaconda or PyTorch, multiple OpenMP DLLs (`libiomp5md.dll`) can conflict, producing fatal error `OMP: Error #15`.

In this suite, this is prevented at the root:
```python
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
```
This environment flag is automatically injected when importing `src`.

---

## 2. Windows Terminal Encoding Safety

Windows PowerShell and CMD traditionally use code page `cp1252`, which raises `UnicodeEncodeError: 'charmap' codec can't encode character '\u2605'` when logging special characters like stars (`★`).

The entire codebase uses standard ASCII character markers (`*`, `[BEST]`, `[OK]`, `[PASS]`) to guarantee crash-free terminal operation across all Windows command interpreters.

---

## 3. PowerShell Execution Policy

If script execution is blocked on your system:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

---

## 4. CUDA GPU Acceleration

To ensure PyTorch utilizes your NVIDIA GPU:
```powershell
python -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0))"
```
The configurations in `configs/*.toml` default to `device = "cuda"`. If no CUDA-capable GPU is found, the pipeline gracefully falls back to `cpu`.
