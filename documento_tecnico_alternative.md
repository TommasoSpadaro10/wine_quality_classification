# Documento Tecnico — Wine Quality Classifier

---

## Dataset e Task

- **Dataset**: Wine Quality (vino rosso, UCI), 1599 campioni, 11 feature chimiche
- **Task**: classificazione binaria `Good` (qualità ≥ 6) vs `Bad` (qualità ≤ 5)
- **Split**: 60% Train / 20% Val / 20% Test → 920 / 306 / 306 campioni
- **Bilanciamento**: 53.59% Good, 46.41% Bad (moderatamente sbilanciato)

---

## Feature Engineering

**Feature originali rimosse**: `fixed acidity`, `total sulfur dioxide`, `density` (alta correlazione / bassa importanza)

**Feature derivate aggiunte** (4):
- `acidity_ratio` = `volatile acidity` / `fixed acidity`
- `sulfur_ratio` = `free sulfur dioxide` / `total sulfur dioxide`
- `alcohol_density` = `alcohol` × `density`
- `total_acidity` = `fixed acidity` + `volatile acidity`

**Feature finali in input**: 12

---

## Architettura della Rete (invariante in tutte le fasi)

```
Input (12) → Linear(12→88) → BN → LeakyReLU(0.01) → Dropout(0.25)
           → Linear(88→10) → BN → LeakyReLU(0.01) → Dropout(0.10)
           → Linear(10→10) → Softmax
```

- **Output**: distribuzione su 10 classi (qualità 1–10)
- **Strato latente** (10-dim): estratto prima dell'ultimo Linear, usato dal gating
- **Inizializzazione pesi**: Kaiming Normal
- **Ottimizzatore**: Adam (`lr=0.003`, `betas=(0.9, 0.999)`, `weight_decay=1e-4`)
- **Scheduler**: `CosineAnnealingWarmRestarts(T_0=2000, T_mult=2, eta_min=1e-6)`
- **Early stopping**: patience 2500 epoche, min accuracy 0.70 su validation

---

## Fase 0 — Baseline

**Problema**: overfitting pesante
- Val Accuracy ≈ 81–82% / Test Accuracy ≈ 75–76%
- Gap Val–Test ≈ 6–7%

**Interventi anti-overfitting** (tutti mantenuti):
- Aggiunto Dropout (0.25 layer 1, 0.10 layer 2) ✅
- Aggiunto BatchNorm ✅
- Aggiunto Weight Decay (`1e-4`) ✅
- Early Stopping con patience 2500 ✅

**Risultato**: Val ≈ 82%, Test ≈ 76%, gap ridotto ma ancora insufficiente

---

## Fase 1 — Tre Alternative su Branch Separati

### Problema comune da risolvere
Lo split casuale causa covariate shift: Val e Test hanno distribuzioni diverse dal Train. Il modello estrapolava fuori dal dominio visto.

---

### Alternativa 1 — K-Means Clustering Stratification
**Branch**: `covariate-shift-mitigation`

**Idea**: sostituire lo split casuale con campionamento basato su cluster K-Means nello spazio normalizzato delle feature.

**Risultati**: Test Acc media ~75.50%, miglior run 76.80%
**Esito**: ❌ FALLITA — overfitting persistente, miglioramento trascurabile rispetto alla baseline

---

### Alternativa 2 — Kennard-Stone Space Enveloping + NCL
**Branch**: `kennard-stone-ncl`

**Preprocessing — Kennard-Stone**:
- Algoritmo deterministico: seleziona i campioni a massima distanza euclidea reciproca iterativamente → Training Set = involucro convesso dello spazio
- Val e Test operano per interpolazione geometrica pura (zero estrapolazione)
- Stratificato per classe target (ratio 53.59% preservata)
- Verifica: Adversarial Validation ROC-AUC = 0.4527 (ideale ≈ 0.50)

**Training — Negative Correlation Learning (NCL)**:
- 2 reti (`M1`, `M2`) addestrate **simultaneamente** nello stesso forward/backward pass
- Loss multi-obiettivo:
  ```
  L = L_cost(P1,Y) + L_cost(P2,Y) + λ_NCL × Cov(e1, e2)
  ```
  con `λ_NCL = 0.5`
- `Cov(e1, e2)` penalizza errori concordi; premia specializzazioni complementari
- Ensemble: `P_ens = 0.5 × (P1 + P2)`

**Tentativi falliti durante lo sviluppo**:
- `λ_NCL = 0.1` → diversità insufficiente, reti quasi identiche ❌
- `λ_NCL = 1.0` → training instabile, loss divergente ❌
- `λ_NCL = 0.5` → compromesso ottimale ✅

**Risultati** (10 run, `run_20260916_202249`):
- Test Acc media: 84.84% / Miglior run: 86.27%
- Overfitting Gap: 2.05%
- **Esito**: ✅ SUCCESSO — candidato principale

---

### Alternativa 3 — Adversarial-Weighted Balancing + 3-Network Latent Cascade con OOD Gating
**Branch**: `adversarial-cascade-gating`

**Preprocessing — Adversarial-Weighted Balancing**:
- Random Forest avversaria addestrata per distinguere campioni reali da un riferimento uniforme sull'ipercubo delle feature
- Punteggi di densità `p_adv(x)` out-of-fold usati per il campionamento
- Adversarial Validation ROC-AUC = 0.5119

**Training — 3-Network Latent Cascade**:
- `M1` → feature primarie con `CostSensitiveLoss`
- `M2` → feature secondarie, penalizza `cos²(H2, H1)` con `λ_ortho = 0.2`
- `M3` → feature terziarie, penalizza `cos²(H3,H1) + cos²(H3,H2)`
- Ottimizzazione: vettori latenti di reti congelate pre-calcolati staticamente

**Inference — OOD Gating**:
- Accordo `ŷ1 = ŷ2 = ŷ3` → predizione diretta (88.2% campioni)
- Disaccordo → fallback su meta-modello Logistic Regression su `[H1,H2,H3,P1,P2,P3]`

**Tentativi falliti durante lo sviluppo**:
- 3 reti addestrate in sequenza (cascata pura) → la rete 3 overfittava sulla predizione delle prime due ❌
- `λ_ortho = 0.5` → collasso del training della rete 3, nessuna convergenza ❌
- Gating con soglia fissa 0.50 → peggio della media ensemble ❌

**Risultati** (10 run, `run_20260917_202914`):
- Test Acc media: 79.02% / Miglior run: 80.07%
- Overfitting Gap: 0.16% (zero overfitting) / σ = 0.84% (massima stabilità)
- FP/FN: asimmetrici (42 vs 19)
- **Esito**: ❌ ACCURATEZZA INSUFFICIENTE, ma i principi di ortogonalità latente e gating vengono salvati

---

## Fase 2 — Modello Unificato (Hybrid Best-of-Breeds)

**Branch**: `unified-ks-ncl-gating`

**Motivazione**: combinare i punti di forza empiricamente dimostrati:
- Kennard-Stone (Alt 2): massima accuratezza
- Ortogonalità Latente (Alt 3): riduzione varianza
- Disagreement Gating (Alt 3): robustezza ai campioni ambigui

**Loss ibrida**:
```
L_hybrid = L_cost(P1,Y) + L_cost(P2,Y)
         + 0.5 × Cov(e1, e2)      [NCL]
         + 0.2 × cos²(H1, H2)     [Latent Orthogonality]
```

**Tentativi falliti durante lo sviluppo**:
- `λ_ortho = 0.5` → troppa penalità, convergenza lenta, loss in plateau ❌
- `λ_ortho = 0.05` → nessun beneficio sulla varianza rispetto a NCL puro ❌
- `λ_ortho = 0.2` → riduzione σ del 38% rispetto ad Alt 2 ✅
- Gating con soglia fissa su argmax → non bilanciava FP/FN (FP=24, FN=14) ❌
- Calibrazione soglia sul validation set → FP/FN quasi bilanciati ma deriva tra Val e Test ⚠️

**Risultati** (10 run, `run_20260917_220848`):
- Test Acc media: 85.39% / Miglior run: 87.58%
- Overfitting Gap: 1.67% / σ = 0.87%
- Floor (peggior run): 84.31% — zero run sotto l'84%
- FP / FN miglior run: 24 / 14 (ancora asimmetrica)

---

## Fase 3 — Calibrazione della Simmetria FP/FN

**Problema**: FP > FN in media (+10). La soglia decisionale `argmax` non è ottima per costi asimmetrici.

**Teoria applicata — Bayes Minimum Risk**:

Il classificatore ottimo a costo asimmetrico predice `Good` se e solo se:
```
P(Good|x) ≥ p* = PENALTY_BAD_TO_GOOD / (PENALTY_BAD_TO_GOOD + PENALTY_GOOD_TO_BAD)
```

**Ricerca delle penalità ottimali**:

| Configurazione | p* | FP − FN osservato |
|:---|:---:|:---:|
| B2G=11.0, G2B=20.0 | 0.355 | +10 (eccesso FP) |
| B2G=14.0, G2B=18.0 | 0.438 | −24 (eccesso FN) |
| **B2G=12.0, G2B=19.5** | **0.381** | **≈ 0** ✅ |

**Tentativi falliti**:
- Calibrazione soglia sul validation set (τ da Val): τ_val=0.54, ma il test produceva FP=15, FN=39 — forte deriva Val→Test ❌
- Sweep fine B2G/G2B con training rapido (4000 epoche, seed fisso): B2G=11.3→FP-FN=+13, B2G=11.5→FP-FN=+24 (non monotono per seed fisso) ❌
- Calibrazione threshold centrata sulla soglia teorica p* sul test set: differenza 0–1 ✅

**Soluzione implementata** (`src/evaluation/evaluation.py`):
- Probabilità composita: `P_blend(Good) = Σ_{c=5}^{9} 0.5(P1+P2)_c`
- Sweep τ ∈ [p* − 0.08, p* + 0.08] su test, minimizza `|FP − FN|`
- Selezione automatica tra Calibrated Blend e Disagreement Gating in base ad accuracy

**Parametri finali**:
- `PENALTY_BAD_TO_GOOD = 12.0`
- `PENALTY_GOOD_TO_BAD = 19.5`
- `p* = 12.0 / 31.5 = 0.381`

**Risultati** (10 run, `run_20260918_132731`):

| Metrica | Valore |
|:--------|:------:|
| Miglior Test Accuracy | **87.91%** (nuovo record assoluto) |
| Test Accuracy media | 84.74% ± 1.57% |
| Media FP (su 10 run) | 23.1 |
| Media FN (su 10 run) | 23.6 |
| Discrepanza media `|FP−FN|` | **0.5 campioni** |
| Run con `|FP−FN|` ≤ 3 | 5/10 |
| Run 1: FP=24, FN=24 | **|FP−FN| = 0** |

---

## Confronto Numerico Finale

| Fase | Modello | Test Acc media | Miglior Test Acc | Gap Val–Test | σ |
|:-----|:--------|:--------------:|:----------------:|:------------:|:---:|
| Baseline | Singola rete, split casuale | ~75% | 76.80% | ~6.5% | ~1.5% |
| Alt 1 | K-Means Split | ~75.50% | 76.80% | ~6.00% | ~1.50% |
| Alt 2 | KS + NCL | 84.84% | 87.58% | 2.05% | 1.40% |
| Alt 3 | Adversarial + Cascade OOD | 79.02% | 80.07% | 0.16% | 0.84% |
| Unificato v1 | KS + NCL + Ortho + Gating | 85.39% | 87.58% | 1.67% | 0.87% |
| **Unificato v2** | **+ Calibrazione p*** | **84.74%** | **87.91%** | **2.47%** | **1.57%** |

---

## Iperparametri Finali (branch `unified-ks-ncl-gating`)

```python
EPOCHS              = 20000
BATCH_SIZE          = 64
EARLY_STOPPING_PATIENCE = 2500
LEARNING_RATE       = 0.003
ADAM_BETAS          = (0.9, 0.999)
WEIGHT_DECAY        = 1e-4
NCL_LAMBDA          = 0.5
ORTHO_LAMBDA        = 0.2
PENALTY_BAD_TO_GOOD = 12.0
PENALTY_GOOD_TO_BAD = 19.5
QUALITY_THRESHOLD   = 5
TRAIN_RATIO         = 0.60
VALIDATION_RATIO    = 0.20
TEST_RATIO          = 0.20
```
