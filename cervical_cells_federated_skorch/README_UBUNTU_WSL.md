# Ubuntu / WSL2 Linux Deployment Guide

This document describes how to deploy and execute the **Flower Federated Learning Suite for Cervical Cytology with skorch** on Ubuntu 22.04 / 24.04 LTS or Windows Subsystem for Linux (WSL2).

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

---

## 2. Quick Setup

Using `pixi`:
```bash
chmod +x setup_pixi.sh
./setup_pixi.sh
```

Or using `uv`:
```bash
chmod +x setup_uv.sh
./setup_uv.sh
```

Or using `venv`:
```bash
chmod +x setup_venv.sh
./setup_venv.sh
```

---

## 3. Headless Execution

```bash
export MPLBACKEND=Agg
python3 main_federated_skorch.py --config configs/default.toml
```
