# Ubuntu & WSL2 Deployment Guide

This guide details configuration on Ubuntu 22.04+ and Windows Subsystem for Linux (WSL2).

---

## 🛠️ System Prerequisites

```bash
sudo apt update
sudo apt install -y build-essential libgl1-mesa-glx libglib2.0-0 git curl
```

---

## 🚀 Execution

```bash
# Using Pixi
pixi run test-dual

# Using standard venv
bash scripts/setup_venv.sh
source .venv/bin/activate
python main_federated_fedml.py --preprocessing n2n_cyclegan
```

All Matplotlib rendering utilizes headless backends (`plt.switch_backend('Agg')` compatible) ensuring seamless generation of PNG plots without X11 server displays.
