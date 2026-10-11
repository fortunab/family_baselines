# Conda / Mamba Environment Reference Guide
## Skin Lesion Federated Learning Suite

Acest ghid oferă instrucțiuni complete pentru gestionarea mediilor cu **Anaconda**, **Miniconda** sau **Mamba**.

---

## 1. Structura Fișierului `environment.yml`

Fișierul `environment.yml` inclus specifică:
- Canale oficiale: `pytorch`, `conda-forge`, `defaults`
- Versiune Python: `3.11`
- Dependențe native și pachete PyPI: PyTorch, torchvision, wandb, scikit-learn, etc.

---

## 2. Creare și Activare Mediu

### Creare Automată din `environment.yml`:
```bash
conda env create -f environment.yml
```

### Activare:
```bash
conda activate skin_lesion_fedml
```

### Dezactivare:
```bash
conda deactivate
```

---

## 3. Comenzi Utile Conda

```bash
# Listare medii existente
conda info --envs

# Exportarea configuratiei curente a mediului
conda env export > environment_lock.yml

# Stergerea mediului (daca doresti resetare completa)
conda env remove -n skin_lesion_fedml
```

---

## 4. Accelerare cu Mamba (Opțional)
Dacă procesul de rezolvare a dependențelor Conda este lent, instalează Mamba:
```bash
conda install mamba -n base -c conda-forge
mamba env create -f environment.yml
```
