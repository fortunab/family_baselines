# Ubuntu / WSL2 Linux Deployment Guide

This document describes how to deploy and execute the **Flower Federated Learning Suite for Cervical Cytology** on Ubuntu 22.04 / 24.04 LTS or Windows Subsystem for Linux (WSL2).

---

## 1. System Prerequisites

Install base system dependencies:
```bash
sudo apt-get update && sudo apt-get install -y \
    build-essential \
    python3-dev \
    python3-venv \
    python3-pip \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    curl
```

### NVIDIA CUDA on WSL2 / Ubuntu
Ensure NVIDIA container toolkit or standard CUDA driver is active:
```bash
nvidia-smi
```

---

## 2. Quick Setup with Shell Script

```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

Or using `uv`:
```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

---

## 3. Headless Ray & Flower Simulation

On headless servers or clusters:
```bash
# Set headless rendering for matplotlib
export MPLBACKEND=Agg

# Launch Flower Simulation
python3 main_federated_flower.py --config configs/convnext.toml
```

---

## 4. Multi-GPU Allocation with Ray
In Flower's simulation engine (`flwr.simulation.start_simulation`), resources are automatically distributed across available GPUs.
To restrict execution to a single GPU on multi-GPU nodes:
```bash
CUDA_VISIBLE_DEVICES=0 python3 main_federated_flower.py --config configs/convnext.toml
```
