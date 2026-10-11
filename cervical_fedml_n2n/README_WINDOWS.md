# Windows 11 Native Execution Guide

This suite is fully optimized for native Windows 11 environments using PowerShell and Command Prompt.

---

## 🛡️ Windows-Specific Hardening

1. **OpenMP Runtime Conflicts**:
   - `os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"` is enforced at package load time in `src/__init__.py` to prevent crashes when both PyTorch and MKL/OpenMP runtimes are present.
2. **Device Resolution**:
   - `resolve_device()` checks `torch.cuda.is_available()`. If CUDA is uncompiled or drivers are absent, it cleanly falls back to CPU without throwing PyTorch assertion errors.
3. **Console Encoding**:
   - All console outputs use ASCII asterisks and characters to prevent Windows `cp1252` encoding errors.
4. **Execution Policy**:
   - If PowerShell scripts are restricted, run:
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
     ```
