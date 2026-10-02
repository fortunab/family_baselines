# "Random Best Select" Optimizer & Strict Elimination of Seed 42

This document details the dynamic seed selection philosophy and multi-candidate entropy search implemented in `src/seed_selector.py`.

---

## 1. Zero Tolerance for Hardcoded Seed 42

Hardcoded random seeds (especially seed 42) introduce partition bias, artificially cherry-pick benchmark splits, and fail to simulate true real-world decentralized hospital variability. 

In this suite:
1. Hardcoded seed 42 is strictly prohibited.
2. If seed 42 is passed via config or CLI, `src/seed_selector.py` intercepts it, prints a security warning, and dynamically replaces it with a cryptographically secure seed.

---

## 2. Cryptographic High-Entropy Seed Generation

Dynamic seeds are generated using Python's `secrets` module:
```python
seed = secrets.randbelow(900000) + 100000
```
This guarantees an unpredictable 6-digit integer in $[100000, 999999]$, seeding Python's `random`, NumPy's `np.random`, and PyTorch's `torch.manual_seed`.

---

## 3. "Random Best Select" Multi-Candidate Search

Rather than relying on an arbitrary random split, the **Random Best Select** optimizer searches for a partitioning that maximizes clinical balance across decentralized hospital silos:

1. **Candidate Generation**: Evaluates $K$ (default $K=5$) independent cryptographic seed candidates.
2. **Shannon Entropy Calculation**: For each client silo $k$ and candidate seed $s$, calculates normalized class distribution entropy:
   $$H_k(s) = -\frac{1}{\log_2(C)} \sum_{c=1}^C p_{k,c} \log_2(p_{k,c})$$
   where $C=7$ classes and $p_{k,c}$ is the proportion of class $c$ in client $k$.
3. **Partition Balance Score**:
   $$\bar{H}(s) = \frac{1}{M} \sum_{k=1}^M H_k(s)$$
4. **Selection**: Selects the candidate seed $s^*$ with the highest mean entropy $\bar{H}(s^*)$, ensuring equitable clinical representation across all hospital silos.

---

## 4. Windows Unicode Console Compatibility

All logging in `src/seed_selector.py` uses standard ASCII formatting (e.g., `*` and `[BEST]`) instead of Unicode stars (`★`), completely avoiding `UnicodeEncodeError` crashes on Windows terminal code pages (`cp1252`).
