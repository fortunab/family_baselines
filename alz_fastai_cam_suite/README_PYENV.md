# `pyenv` Setup Guide: Alzheimer MRI CAM Suite

Instructions for managing Python versions and isolated environments using `pyenv` (Linux/macOS/WSL) and `pyenv-win` (Windows).

---

## 1. Version Pinning
This project defines the exact Python version in `.python-version`:
```
3.11.9
```

---

## 2. Windows (`pyenv-win`)

### Installation
If `pyenv-win` is not installed:
```powershell
Invoke-WebRequest -UseBasicParsing -Uri "https://raw.githubusercontent.com/pyenv-win/pyenv-win/master/pyenv-win/install-pyenv-win.ps1" -OutFile "./install-pyenv-win.ps1"; &"./install-pyenv-win.ps1"
```

### Environment Initialization
```powershell
# 1. Install Python 3.11.9
pyenv install 3.11.9

# 2. Set local version for project
pyenv local 3.11.9

# 3. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 3. Linux / Ubuntu WSL (`pyenv`)

### Installation & Build Dependencies
```bash
sudo apt update
sudo apt install -y make build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev wget curl llvm \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

# Install pyenv via installer script
curl https://pyenv.run | bash
```

Add pyenv to your `~/.bashrc`:
```bash
export PYENV_ROOT="$HOME/.pyenv"
[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"
eval "$(pyenv init -)"
```

### Environment Initialization
```bash
# 1. Install Python 3.11.9
pyenv install 3.11.9

# 2. Set local version
pyenv local 3.11.9

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Execution
```bash
python visualize_cam_heatmaps.py
```
