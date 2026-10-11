# Federated Learning (FedML / FedProx) & Weights & Biases (W&B) Architecture
## Clinical Dermatology Dermoscopy Benchmark

Acest document explică fundamentele matematice și arhitecturale ale sistemului federat implementat pentru clasificarea a 7 tipuri de leziuni dermatologice pe cohorta multi-instituțională HAM10000 / ISIC.

---

## 1. Arhitectura de Învățare Federată (Client-Server Topology)

În mediul clinic real, imaginile dermatoscopice ale pacienților nu pot fi partajate direct între spitale din cauza reglementărilor stricte de confidențialitate (GDPR, HIPAA). Învățarea federată permite colaborarea spitalelor prin partajarea exclusivă a **ponderilor modelului**, păstrând datele pacienților strict locale:

```
                  +-----------------------------------+
                  |      CENTRAL FEDERATED SERVER     |
                  |  - Distribuie Consensus w_t       |
                  |  - Agregare Ponderi (FedAvg/Prox) |
                  |  - Evaluare Cohorta Globala Test  |
                  +-----------------+-----------------+
                                    |
          +-------------------------+-------------------------+
          |                         |                         |
+---------v---------+     +---------v---------+     +---------v---------+
| Hospital Vienna   |     | Sydney Melanoma   |     | MSK New York      |
| Local Data (Priv) |     | Local Data (Priv) |     | Local Data (Priv) |
| Local SGD/AdamW   |     | Local SGD/AdamW   |     | Local SGD/AdamW   |
+-------------------+     +-------------------+     +-------------------+
```

---

## 2. Simularea Distribuției Non-IID Clinice (Dirichlet Partition $\alpha$)

Fiecare centru medical are o prevalență diferită a subtipurilor de leziuni (de exemplu, un centru specializat în melanom va avea o proporție disproporționat de mare de cazuri `MEL` comparativ cu o clinică generală de dermatologie).

Această eterogenitate este modelată matematic prin distribuția **Dirichlet** cu parametrul de concentrație $\alpha$:
$$\mathbf{p}_c \sim \text{Dir}(\alpha \cdot \mathbf{1}_M)$$
unde $M$ este numărul de clienți (spitale), iar $c \in \{0, \dots, 6\}$ reprezintă cele 7 clase.
- $\alpha = 0.5$: Dezechilibru ridicat (heterogenitate clinică realistă / Non-IID extrem).
- $\alpha = 10.0$: Distribuție uniformă IID.

---

## 3. Algoritmi Federate: FedAvg vs. FedProx

### A. Federated Averaging (FedAvg - McMahan et al.)
Fiecare spital $k$ antrenează modelul local timp de $E$ epoci, iar serverul calculează media ponderată:
$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_k^{t+1}$$
*Limitare*: În condiții Non-IID severe, optimizarea locală diverghează puternic de consensul global ("client drift").

### B. Federated Proximal (FedProx - Li et al.)
Pentru a preveni divergența clienților în cazul etichetelor Non-IID, FedProx adaugă un termen de regularizare proximală la funcția de pierdere a fiecărui client:
$$\mathcal{L}_{\text{FedProx}}(w) = \mathcal{L}_{\text{task}}(w) + \frac{\mu}{2} \|w - w_t\|^2$$
- $w_t$: Modelul de consens global primit de la server la începutul rundei.
- $\mu$: Pondere proximală (ex. $\mu = 0.01$). Penalizează deviațiile exagerate ale spitalului de la modelul comun.

---

## 4. Integrarea Weights & Biases (W&B)

Sistemul suportă două moduri native:

### Mod Offline (`mode = "offline"` - Implicit)
- Nu necesită cont W&B și nici conexiune la internet.
- Salvează toate metricile, rundele și artefactele pe disc local (`wandb/offline-run-...`).
- Se poate sincroniza ulterior oricând cu:
  ```bash
  wandb sync wandb/offline-run-*
  ```

### Mod Online (`mode = "online"`)
- Transmite metricile în timp real către dashboard-ul online W&B.
- Urmărește:
  - `global/accuracy`, `global/balanced_accuracy`, `global/cohen_kappa`, `global/macro_f1`
  - `global/referral_sensitivity` (siguranța pacientului pentru leziuni maligne)
  - `client_{id}/train_loss`, `client_{id}/divergence`
  - Artefacte de imagini și puncte de salvare ale modelului (`.pt`).

---

## 5. Sistem Dublu de Configurare: TOML și INI (`configparser`)

1. **TOML (`configs/federated_config.toml`)**:
   - Format modern, suport nativ în Python 3.11+ via `tomllib` (sau `tomli`).
   - Suportă tipuri de date native (liste, dicționare, numere reale, booleene).
2. **INI (`configs/federated_config.ini`)**:
   - Format clasic utilizat pe scară largă în industrie, compatibil 100% cu `configparser` din biblioteca standard Python.
   - Încărcătorul unificat `src/config_loader.py` detectează automat extensia și mapează valorile în dataclass-ul `FederatedConfig`.
