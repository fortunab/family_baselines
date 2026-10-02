# Dynamic High-Entropy Seeds & "Random Best Select" Optimizer

## Elimination of Hardcoded Seed 42

In accordance with strict benchmark guidelines, **seed 42 is completely prohibited** across all experiment runners, partitioners, and configuration profiles:
1. **Zero Hardcoded Seeds**: All default configurations (`configs/*.toml`) specify `seed = "random_best"`.
2. **Strict Guardrail**: If seed 42 is passed via `--seed 42`, `src/seed_selector.py` intercepts the call, displays an alert, and substitutes it with a high-entropy cryptographically generated random seed.

---

## Cryptographic Entropy Generation

Seeds are drawn using Python's `secrets` module (operating-system entropy source):
```python
seed = secrets.randbelow(900000) + 100000  # Cryptographic 6-digit integer in [100000, 999999]
```

---

## "Random Best Select" Algorithm

When running `main_federated_substra.py`, the system automatically activates the **Random Best Select Optimizer**:

```
                    +-----------------------------+
                    | Sample 5 Cryptographic      |
                    | Candidate Seeds (e.g. 5x)   |
                    +--------------+--------------+
                                   |
                                   v
             +---------------------+---------------------+
             |  Simulate Node Partitioning for each Seed  |
             +---------------------+---------------------+
                                   |
                                   v
           +-----------------------+-----------------------+
           | Compute Shannon Entropy & Distribution Score |
           +-----------------------+-----------------------+
                                   |
                                   v
             +---------------------+---------------------+
             |  Select Top-Ranked Candidate (* WINNER)   |
             +---------------------+---------------------+
                                   |
                                   v
             +---------------------+---------------------+
             |  Broadcast Best Seed to Substra DAG Nodes |
             +-------------------------------------------+
```

### Mathematical Scoring Function:
$$\text{Score}(s) = 0.5 \cdot \bar{H}(P) + 0.3 \cdot \text{Coverage}_{\text{test}} - 0.2 \cdot \text{Penalty}_{\text{size}}$$

Where:
- $\bar{H}(P) = \frac{1}{N} \sum_{c=1}^N \frac{-\sum_{k=1}^7 p_{c,k} \log_2 (p_{c,k})}{\log_2(7)}$ is the average normalized class entropy across hospital nodes.
- $\text{Coverage}_{\text{test}}$ verifies all 7 cervical cytology classes are represented in the centralized holdout test partition.
- $\text{Penalty}_{\text{size}}$ penalizes severe sample count imbalances between nodes.

---

## Example Terminal Leaderboard Output

```
======================================================================
    SUBSTRA 'RANDOM BEST SELECT' DYNAMIC SEED OPTIMIZER (NO SEED 42)
======================================================================
  Evaluating 5 cryptographic candidate seeds for optimal distribution...

  Candidate Seed Evaluation Leaderboard:
   #1 Seed: 224758 | Quality Score: 0.7822 | Test Set: 9 | Clients: [16, 16, 15] * [BEST SELECTED]
   #2 Seed: 795217 | Quality Score: 0.7638 | Test Set: 9 | Clients: [16, 16, 15]
   #3 Seed: 251727 | Quality Score: 0.7445 | Test Set: 9 | Clients: [16, 16, 15]
   #4 Seed: 122949 | Quality Score: 0.7437 | Test Set: 9 | Clients: [16, 16, 15]
   #5 Seed: 459500 | Quality Score: 0.7351 | Test Set: 9 | Clients: [16, 16, 15]
======================================================================
```
