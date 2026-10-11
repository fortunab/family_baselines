# Federated Learning (FedML / FedProx) & Weights & Biases (W&B) Architecture
## Multi-Center Diabetic Retinopathy Benchmark

Acest document explică fundamentele matematice și arhitecturale ale sistemului federat pentru clasificarea celor 5 stadii de retinopatie diabetică (ICDR) pe cohorte multi-centru (EyePACS, Messidor-2, IDRiD, APTOS).

---

## 1. Topologia Rețelei Federate

Datele imagistice de fundus de ochi rămân stocate în mod privat pe serverele fiecărei clinici oftalmologice:
- Client 1: `TeleScreening_EyePACS_California` (Camere mobile tele-screening)
- Client 2: `Hospital_Topcon_Paris_Messidor` (Camere non-midriatice de spital terțiar)
- Client 3: `RetinaInstitute_IDRiD_India` (Institut de referință retiniană)
- Client 4: `PrimaryCare_APTOS_Asia` (Clinici primare)

Fiecare centru calculează doar actualizările locale de gradient/ponderi și transmite state-dict-ul către serverul central.

---

## 2. Formularea Matematică FedProx

La runda $t$, clientul $k$ rezolvă problema de optimizare:
$$\min_w \mathcal{L}_k(w; w_t) = \mathcal{L}_{\text{task}}(w) + \frac{\mu}{2} \|w - w_t\|_2^2$$
unde $\mu = 0.01$ este factorul proximal ce limitează deviația locală față de consensul global $w_t$.

Serverul central execută agregarea ponderată:
$$w_{t+1} = \sum_{k=1}^K \frac{n_k}{N} w_{k, t+1}$$

---

## 3. Evaluarea Clinică: Quadratic Weighted Kappa ($\kappa_w$, QWK)

În retinopatia diabetică, stadiile bolii sunt ordinale:
$0 = \text{Normal} \to 1 = \text{Mild} \to 2 = \text{Moderate} \to 3 = \text{Severe} \to 4 = \text{Proliferative}$.

Confundarea stadiului 0 cu 1 este o eroare minoră, în timp ce confundarea stadiului 0 cu 4 este o eroare clinică catastrofală.
Quadratic Weighted Kappa penalizează matricea de confuzie pătratic:
$$w_{i,j} = \frac{(i-j)^2}{(K-1)^2}$$
Pragul pentru utilizare clinică autonomă în oftalmologie este $\kappa_w \ge 0.80$ (acord aproape perfect).

---

## 4. Integrarea Weights & Biases (W&B)

Sistemul suportă:
- **Mod Offline** (`mode = "offline"`): Complet autonom, fără cont și fără internet, salvând local metricele fiecărei runde.
- **Mod Online** (`mode = "online"`): Sincronizare automată în cloud pentru tablouri de bord interactive.
- Urmărește: `global/accuracy`, `global/quadratic_weighted_kappa`, `global/referral_sensitivity` ($\ge 95\%$), `client/divergence` și salvează artefacte de imagini.
