# Ubuntu Linux & WSL2 Execution Guide

This guide covers deploying the FedML Cervical Cytology suite on Ubuntu Linux and Windows Subsystem for Linux (WSL2).

---

## 1. System Prerequisites

Ensure essential build tools and Python 3.11 development libraries are present:
```bash
sudo apt update
sudo apt install -y build-essential curl git python3-dev python3-pip python3-venv
```

---

## 2. NVIDIA CUDA Drivers on WSL2

For WSL2 users with NVIDIA GPUs:
1. Ensure the latest NVIDIA GPU driver is installed on Windows host.
2. Inside WSL2, verify GPU access:
   ```bash
   nvidia-smi
   ```

---

## 3. Quick Deployment via Bash Scripts

Execute any of the automated bootstrap scripts:
```bash
# Using Pixi
chmod +x scripts/setup_pixi.sh && ./scripts/setup_pixi.sh

# Using uv
chmod +x scripts/setup_uv.sh && ./scripts/setup_uv.sh

# Using standard venv
chmod +x scripts/setup_venv.sh && ./scripts/setup_venv.sh
```

---

## 4. Headless Execution

When running on headless Linux servers, matplotlib figures are generated using headless backends (`Agg`), completely avoiding DISPLAY variable errors.
