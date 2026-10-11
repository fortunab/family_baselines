# Random Best Select Seed Optimizer & Policy

This module enforces a strict policy against arbitrary hardcoded seeds (such as seed 42) in favor of dynamic cryptographic randomness and multi-candidate entropy optimization.

---

## 🚫 Prohibition of Seed 42

Seed 42 is explicitly intercepted in `src/seed_selector.py`:
- If `--seed 42` is passed, a warning is printed and a dynamic high-entropy cryptographic seed is automatically substituted via `secrets.randbelow(900000) + 100000`.

---

## 🎲 Multi-Candidate Random Best Select

Instead of relying on a single lucky split, the system generates $K$ independent cryptographic seed candidates and evaluates their partition balance using **Shannon Class Distribution Entropy**:

$$\mathcal{H}(S_k) = -\sum_{c=1}^C p_c \log_2(p_c)$$

The seed candidate producing the highest mean entropy across all hospital client silos is automatically selected as the optimal data split seed.

---

## 💻 CLI Usage

```powershell
# Default: 5 candidate seeds evaluated, best chosen
python main_federated_fedml.py

# Evaluate 10 candidates
python main_federated_fedml.py --candidates 10

# Single dynamic cryptographic seed (skipping multi-candidate search)
python main_federated_fedml.py --no_random_best_select

# Explicit valid user seed (e.g. 841612)
python main_federated_fedml.py --seed 841612
```
