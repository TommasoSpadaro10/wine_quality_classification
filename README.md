# 🍷 Wine Quality Classification
> *Considerazioni che mi hanno portato all'87.91%.*  
> 🌐 **[Read this in English](README_en.md)**

È noto che per affrontare problemi con dati tabellari non molto estesi, si tende ad usare le solite tecniche di machine learning "avanzato" come XGBoost, LightGBM, ecc, piuttosto che reti neurali, che tendono ad andare subito in overfitting o la scarsità di dati non permette di arrivare a performance accettabili. 

Da qui, la sfida personale: ho voluto provare a costruire una rete neurale che potesse competere con le tecniche avanzate di machine learning. Provando con XGBoost ho ottenuto un'accuratezza del 83% circa. 

Spoiler: sono riuscito ad ottenere un'accuratezza del **87.91%** su test set, con equilibrio tra Falsi Positivi e Falsi Negativi. 
Sono partito con un'idea: sfruttare a mio favore la limitatezza del data set e la semplicità della struttura del problema. 

---

## Mappa del Percorso

- [Quickstart](#quickstart)
- [Metodologia: Split dei Dati e Tecniche di Training](#metodologia-split-dei-dati-e-tecniche-di-training)
- [1. Il Problema](#1-il-problema)
- [2. Le Mie Due Intuizioni](#2-le-mie-due-intuizioni)
- [3. I Vicoli Ciechi e lo stallo](#3-i-vicoli-ciechi-e-lo-stallo)
- [4. Superamento dello stallo: Kennard-Stone, NCL e Joint Training](#4-superamento-dello-stallo-kennard-stone-ncl-e-joint-training)
- [5. I Risultati Finali e il Record dell'87.91%](#5-i-risultati-finali-e-il-record-dell8791)
- [6. Tabellone di Tutti gli Esperimenti](#6-tabellone-di-tutti-gli-esperimenti)
- [7. Struttura del Codice e Riproducibilità](#7-struttura-del-codice-e-riproducibilità)

---

## Quickstart

### 1. Clona e installa le dipendenze
```bash
git clone https://github.com/TommasoSpadaro10/wine_quality_classification.git
cd wine_quality_classification

# Con supporto CUDA (opzionale se hai una GPU Nvidia):
pip install --extra-index-url "https://download.pytorch.org/whl/cu126" -r requirements.txt
```

### 2. Esegui la suite di validazione statistica (10 run indipendenti)
```bash
python run_experiments.py
```

---

## Metodologia: Split dei Dati e Tecniche di Training

Prima di addentrarmi nei singoli esperimenti, è utile definire i principi metodologici con cui sono stati gestiti i dati e l'arsenale di tecniche di addestramento adottate "sotto il cofano" per garantire convergenza, stabilità ed evitare l'overfitting.

### 1. I Fondamenti dello Split: Training, Validation e Test Set

La suddivisione di un dataset supervisionato risponde al principio di **generalizzazione su dati non visti**. Nel mio setup i dati vengono ripartiti secondo le proporzioni **60% Train / 20% Validation / 20% Test**:

```text
                       DATASET COMPLETO (1.599 campioni)
   ┌──────────────────────────────────┬─────────────────┬─────────────────┐
   │        TRAINING SET (60%)        │ VALIDATION (20%)│    TEST (20%)   │
   │           ~959 campioni          │  ~320 campioni  │  ~320 campioni  │
   └──────────────────────────────────┴─────────────────┴─────────────────┘
                   │                           │                 │
       Backpropagation & Pesi       Tuning Iperparametri,   Valutazione
       (Aggiornamento W e b)         Early Stopping,         Finale Cieca
                                     Soglie Bayesiane        (Unbiased)
```

#### A. Ruoli e Separazione dei Compiti
1. **Training Set (60%)**:
   - È l'unico insieme su cui la rete calcola i gradienti con la backpropagation e aggiorna i pesi $W$ e i bias $b$.
2. **Validation Set (20%)**:
   - **Nessun gradiente viene calcolato su questo set.**
   - Utilizzato per:
     - Monitorare l'insorgenza dell'overfitting.
     - Guidare l'**Early Stopping**.
3. **Test Set (20%)**:
   - Viene valutato **una sola volta alla fine** per misurare le reali capacità della rete.

#### B. Split 
- **Random Split Standard**: suddivide i dati casualmente.
- **Stratified Split**: vincola il campionamento per mantenere identica la percentuale di ciascuna classe nei tre insiemi.

---

### 2. Tecniche di Training Adottate "Sotto il Cofano"

Oltre alla formulazione della funzione di costo Cost-Sensitive, l'addestramento della rete fa uso di una serie di tecniche che ho mantenuto lungo tutto il progetto poiché di poco interesse(poca creatività a riguardo). Ho scelto tecniche e valori abbastanza standard.

#### A. Inizializzazione dei Pesi (He / Kaiming Normal)
I pesi delle connessioni lineari non sono lasciati al default casuale o a normali generiche, ma vengono inizializzati secondo il criterio **Kaiming Normal** (He Initialization):
$$W \sim \mathcal{N}\left(0, \frac{2}{(1 + a^2) \cdot \text{fan\_in}}\right)$$
con pendenza negativa $a = 0.01$ tarata per la funzione **LeakyReLU**, e bias azzerati. Questo garantisce che la varianza dei segnali rimanga stabile lungo tutti i layer.

#### B. Normalizzazione e Regolarizzazione Multi-Livello
Per rendere il training stabile ho adottato:
1. **Batch Normalization (`BatchNorm1d`)**: mantiene medie e varianze controllate all'interno di ciascun mini-batch, smorzando l'internal covariate shift tra i layer e fungendo da regolarizzatore naturale.
2. **Dropout Differenziato**: $p = 0.25$ sul primo layer intermedio (88 neuroni): disattiva casualmente il 25% delle connessioni per impedire la co-dipendenza tra feature chimiche correlate. $p = 0.10$ sul secondo layer (10 neuroni): applica un filtro più leggero per non destabilizzare la stima delle probabilità prima dell'output.
3. **Weight Decay (Regolarizzazione $L_2$)**: Fissato nell'ottimizzatore a $\lambda = 10^{-4}$. Introduce una penalità proporzionale alla norma quadratica dei pesi ($\frac{1}{2} \lambda \|W\|_2^2$), inducendo la rete a distribuire il carico su coefficienti contenuti ed eliminare pesi sproporzionati.

#### C. Ottimizzazione e Scheduling Dinamico del Learning Rate
- **Ottimizzatore Adam**: Utilizza momenti del primo ordine ($\beta_1 = 0.9$) e secondo ordine ($\beta_2 = 0.999$) con learning rate di partenza $\eta_0 = 3 \times 10^{-3}$.
- **Cosine Annealing with Warm Restarts (`CosineAnnealingWarmRestarts`)**:
  - Il learning rate non è costante, ma segue cicli periodici a coseno:
    $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$
  - Con parametri $T_0 = 2000$ epoche e fattore moltiplicativo $T_{\text{mult}} = 2$, il learning rate decade progressivamente fino a $10^{-6}$ per poi "ripartire".
  - **Perché questa scelta?** Gli sbalzi periodici permettono all'ottimizzatore di sfuggire da minimi locali stretti (spesso soggetti a overfitting) e stabilizzarsi in **flat minima** (minimi piatti e ampi), noti in letteratura per offrire una capacità di generalizzazione nettamente superiore.

#### D. Early Stopping con Soglia Minima
L'addestramento prevede fino a 20.000 epoche potenziali, regolate da un meccanismo di arresto mirato:
- **Patience = 2.500 epoche**: se per 2.500 epoche consecutive l'accuratezza sul Validation Set non migliora, l'addestramento si arresta.
- **Soglia di Attivazione (`EARLY_STOPPING_MIN_ACCURACY = 0.70`)**: l'early stopping entra in azione solo dopo che il modello ha superato il 70% di accuratezza sul Validation Set, evitando interruzioni premature durante la fase esplorativa iniziale.
- **Ripristino del Miglior Modello**: terminato il ciclo, vengono scartati i pesi dell'ultima epoca (spesso affetti da degradazione locale) e vengono ricaricati automaticamente i pesi corrispondenti al picco assoluto registrato sul Validation Set.

---

## 1. Il Problema

Il dataset contiene **1.599 bottiglie** di vino rosso e ognuna ha 11 parametri chimici e un voto dato da sommelier umani da **3 a 8**.


Nel primo tentativo ero partito con l'idea più semplice: impostare una classificazione binaria usando una rete minima:
`Input(11) -> Linear(2) -> Linear(2) -> Output(2)` con Cross-Entropy standard.

Risultato: il modello si fermava a circa il **74-75%** di accuratezza.

In primis quindi mi sono interessato ad ampliare il numero di variabili in ingresso.
Analizzando i grafici di correlazione e facendo alcune speculazioni( ad esempio il grado alcolico e la densità influiscono in modo preponderante sulla qualità del vino), ho rimosso 3 variabili e generate 4 nuove feature, usando quelle rimosse:

$$\text{acidity\_ratio} = \frac{\text{volatile acidity}}{\text{fixed acidity}} \quad \text{(misura lo spunto acetico rispetto alla struttura)}$$

$$\text{sulfur\_ratio} = \frac{\text{free sulfur dioxide}}{\text{total sulfur dioxide}} \quad \text{(frazione attiva contro l'ossidazione)}$$

$$\text{alcohol\_density} = \text{alcohol} \times \text{density} \quad \text{(proxy del corpo idroalcolico)}$$

$$\text{total\_acidity} = \text{fixed acidity} + \text{volatile acidity} \quad \text{(bilancio acido complessivo)}$$

Risultato: lo spazio in ingresso è passato a **12 feature**.

Proseguendo ho supposto che i problemi fossero:
1. **Collo di bottiglia iniziale**: comprimere 11 variabili complesse in appena 2 neuroni fin dal primo layer distruggeva gran parte dell'informazione geometrica dei dati.
2. **La Cross-Entropy non coglie la distanza ordinale**: trattare il problema come una classificazione categorica standard fa sì che ogni errore venga pesato allo stesso modo.

Osservando i grafici di confronto tra le variabili e i grafici feature_vs_target ho supposto che lo spazio delle feature non fosse facilmente separabile nelle due classi. Quindi dovevo dare maggiore libertà alla rete, ossia ampliare la dimensione dei layer così che la rete potesse "imparare" a fare operazioni geometriche più complesse, come rotazioni, scaling, ecc, per separare al meglio le due classi. Sfruttando la piccola quantità di dati, ho fatto vari tentativi e mi sono attestato alla struttura:
`Input(12) -> Linear(88) -> Linear(10) -> Output(10)` con Cross-Entropy standard.

---

## 2. Le Mie Due Intuizioni 

Per fare un salto di qualità dovevo cambiare il modo in cui il modello calcolava la perdita e formulava la predizione.

### A. Riformulazione a 10 classi e CostSensitiveLoss
Invece di forzare la rete a predire subito 0 o 1, ho impostato l'output su **10 classi** (la scala dei voti da 1 a 10), ogni classe i avrà associato un P_i con $$\sum_{i=1}^{10} P_i = 1$$. Di fatto le P_i saranno le probabilità associate ad ogni voto.

Prendendo spunto da "Neural Networks and Deep Learning" di Michael Nielsen ho supposto di dover creare una metrica  semplice e che rispecchiasse un'ipotetico apprendimento umano.
Pertanto, ho costruito una funzione di costo su misura:
$$\mathcal{L}_{\text{cost}} = \sum_{i=1}^{10} P_i \cdot \Big( \vert i - y_{\text{true}} \vert + \text{Penalty}(i, y_{\text{true}}) \Big)$$

Le idee di base sono:
1. **Penalità lineare di distanza**: se un vino ha voto reale 6 e la rete predice 5, è un piccolo errore ($|5 - 6| = 1$); ma se predice 3, la rete ha commesso un errore ben più grande ($|3 - 6| = 3$). L'errore scala linearmente con la distanza, fornendo alla rete un gradiente direzionale continuo (ho provato ad usare il logaritmo della distanza o il quadrato ma ho avuto un peggioramento delle prestazioni).
2. **Penalità cross-threshold**: importante inoltre, se la predizione scavalca il confine tra vino cattivo e buono ($5 \leftrightarrow 6$), viene applicata una sanzione per i falsi positivi e i falsi negativi. Di fatto l'obiettivo finale è quello di valutare correttamente il vino tra buono o cattivo.
3. **Label Smoothing (10%)**: una quota del 10% di costo uniforme evita che la rete diventi iper-sicura e produca logit estremi su campioni rumorosi (dopotutto, i giudizi dei sommelier contengono una quota naturale di soggettività).

### B. Valore Atteso continuo anziché ArgMax
Usare l'`argmax` creava salti bruschi e instabilità, soprattutto sui campioni a cavallo del confine critico 5 vs 6. Ho scelto quindi di calcolare il **Valore Atteso continuo**:
$$\mathbb{E}[\text{quality}] = \sum_{i=1}^{10} i \cdot P(y = i \mid x)$$

In questo modo ottengo un punteggio continuo che riflette anche il grado di incertezza della rete: un vino a cavallo riceve un valore continuo attorno a 5.5 anziché subire una decisione netta e binaria.
 
Ho pure provato ad usare distribuzioni di prababilità più complesse, pensando che una Weibull o una semplice Gaussiana potesse darmi risultati migliori ma non è stato così. Questo sarà dovuto ad un bias, la valutazione dei vini da parte di un umano può seguire queste distribuzioni ma la rete impara sulla base di valori numerici

### C. Calcolo analitico della soglia decisionale (Youden Index)
Per stabilire la soglia di taglio sul valore atteso continuo non ho tirato a indovinare un valore arbitrario (come 5.5). Ho calcolato la curva ROC sul validation set ($\text{AUC} = 0.8317$) e cercato analiticamente il punto che massimizza l'indice $J$ di Youden ($J = \text{TPR} - \text{FPR}$). Il calcolo ha individuato esattamente la soglia **5.0000**. Non una grande scoperta.

Con queste due intuizioni, la singola rete è salita da ~75% a **oltre l'84%**.

---

## 3. I Vicoli Ciechi e lo stallo

Nel percorso ho testato anche idee che sulla carta sembravano promettenti, ma che sul campo si sono rivelate dei vicoli ciechi:

### L'illusione del finto record all'86.64%
A un certo punto, provando a spostare la soglia decisionale a 5.65, ho visto l'accuratezza schizzare all'**86.64%**. Per un momento ho pensato di aver trovato la configurazione perfetta! Poi ho controllato la matrice di confusione e ho scoperto il trucco: avendo alzato la soglia oltre 5.65, avevo escluso tutti i voti 6 dalla classe dei buoni. Il dataset era diventato fortemente sbilanciato e la rete stava semplicemente predicendo quasi sempre "Bad". Un'accuratezza alta solo sulla carta, senza aver imparato nulla di utile.

### I tentativi con architetture Mixture of Experts (MoE)
Ho provato a combinare due reti (un modello A generalista e un modello B specializzato sugli errori di A) guidate da una meta-rete( o regressione logistica) di gating:
* **MoE v1 (Entropy Gate)**: si è fermato a circa l'80% di accuratezza su test set.
* **MoE v2 (Gate con feature chimiche)**: passando le feature in ingresso direttamente, come una sorta di self-attention, al gate ho riscontrato un evidente **overfitting** (83% in validazione ma crollo al 78.5% su test). Su circa 1.600 campioni, una meta-rete finisce per memorizzare i singoli dati invece di imparare una reale strategia di combinazione generale.
Oppure succedeva che la meta-rete tenesse conto solo della prima rete, perché più affidabile nel maggior numero dei casi. Insomma, non sono riuscito a combinare due/tre reti in modo efficace. 

---

## 4. Superamento dello stallo: Kennard-Stone, NCL e Joint Training

Ottimizzata la singola rete, persisteva un problema: un gap costante di circa il 6% tra validazione (~82%) e test (~76%).

Da qui ho supposto che la rete si ritrovasse spesso a estrapolare e non a interpolare i dati nel test set. Con circa 1.500 campioni distribuiti in 12 dimensioni, uno split puramente casuale o stratificato standard lasciava inevitabilmente zone scoperte nel training set. Il modello si trovava così a dover compiere una vera e propria **estrapolazione** geometrica sui dati di test mai visti.

```
      Split Casuale (Problema)              Split Kennard-Stone (Soluzione)
   ┌───────────────────────────┐         ┌───────────────────────────┐
   │  ● Train   ○ Test         │         │  ● Train   ○ Test         │
   │       ●    ●              │         │  ●                     ●  │
   │    ●           ○          │         │       ○    ●    ○         │
   │  ●         ●              │         │    ●    ○    ●     ●      │
   │       ●                   │         │  ●                     ●  │
   └───────────────────────────┘         └───────────────────────────┘
   Il test richiede estrapolazione        Il train occupa l'involucro esterno:
                                          il test opera per interpolazione
```

### A. L'intuizione geometrica: Stratified Kennard-Stone
Per risolvere questo problema alla radice, verso la fine del progetto ho sostituito il campionamento tradizionale con uno **Stratified Kennard-Stone**:
1. **Selezione geometrica per classe**: l'algoritmo opera iterativamente selezionando i campioni a massima distanza euclidea reciproca all'interno di ciascuna classe. Questo garantisce simultaneamente il bilanciamento delle classi (stratificazione) e la massima copertura dello spazio chimico a 12 feature.
2. **Involucro convesso per il Training Set**: popolando il training set (60%) con i campioni estremi e perimetrali, i set di Validation e Test si ritrovano racchiusi all'interno, operando per **pura interpolazione geometrica**.
3. **Verifica anti-shift con Adversarial Validation (5-Fold RF)**: per accertarmi dell'effettiva indistinguibilità statistica tra Train e Validation, ho impiegato un'**Adversarial Validation** basata su K-Fold (5-Fold Stratificato con Random Forest) addestrata a discriminare l'insieme di provenienza del dato. Il risultato ha confermato la -suppongo- assenza di covariate shift ($\text{ROC-AUC} = 0.4527$, valore prossimo al caso puramente casuale 0.50).

### B. Negative Correlation Learning (NCL) con Ortogonalità Latente
Risolto il campionamento, rimaneva il semplice problema di migliorare le performance.
Avendo fatto vari tentativi tramite una rete ensemble o una regressione logistica che prendesse in input i voti di modelli preallenati, di fatto ritrovavo le prestazioni corrispondevano a quelle dei modelli usati in input. Ho pure provato ad usare un metodo Mixture of Experts, addestrando un secondo esperto sulla base degli errori del primo, ma senza successo. 
Da qui ho pensato di forzare l'addestramento di due reti in parallelo in modo che queste "mirassero" all'obiettivo in modo "ortogonale". Uso il termine ortogonale proprio perché utilizzo il valore di ortogonalità dei logit delle due reti ai fini del training. Perciò ho adottato il **Negative Correlation Learning**: un'unica formulazione di costo che penalizza la correlazione tra i residui d'errore dei due modelli e vi aggiunge un vincolo esplicito di **ortogonalità sui vettori latenti** ($H_1 \perp H_2$):

$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{cost}}(P_1, Y) + \mathcal{L}_{\text{cost}}(P_2, Y) + \lambda_{\text{NCL}} \cdot \text{Cov}(e_1, e_2) + \lambda_{\text{ortho}} \cdot \cos^2(H_1, H_2)$$

In questo modo ciascuna rete è costretta a specializzarsi su aspetti complementari dei dati chimici. Questo vincolo ha ridotto la varianza tra run differenti del 38% ($\sigma$ scesa da 1.40% a 0.87%), stabilizzando le performance medie sopra l'85%.

L'idea è che così le due reti entrino in "concorrenza" ma nell'imparare a estrarre due pattern diversi. Forse si potrebbe provare ad allenare più reti con questo metodo, ma suppongo ci sia un limite ai gradi di ortogonalità secondo cui allenare più reti in modo proficuo, questo numero di gradi sarà matematicamente una cifra inferiore rispetto al numero di feature di ingresso. 

### C. Joint Hybrid Training (Addestramento Congiunto)
Quindi ho implementato una **backpropagation congiunta**:
- Un unico passaggio dell'ottimizzatore (`optimizer.step()`) aggiorna all'unisono entrambi i modelli a ogni mini-batch.
- Questo costringe dinamicamente i vettori latenti a divergere geometricamente ($H_1 \perp H_2$) e i loro residui d'errore a decorrelarsi durante l'intero arco dell'addestramento.
---

## 6. I Risultati Finali e il Record dell'87.91%

### Calibrazione Bayesiana della Soglia
Nella prima versione del modello unificato le performance erano ottime, ma persisteva una tendenza a commettere più Falsi Positivi che Falsi Negativi (+10 FP nella miglior run).

Riprendendo la **Teoria Bayesiana del Rischio Minimo**, ho calcolato la soglia teorica ottimale $p^*$ in funzione dei costi asimmetrici:

$$p^* = \frac{\text{PENALTY\_BAD\_TO\_GOOD}}{\text{PENALTY\_BAD\_TO\_GOOD} + \text{PENALTY\_GOOD\_TO\_BAD}} = \frac{12.0}{12.0 + 19.5} = 0.3810$$

Calibrando la soglia $\tau$ nell'intorno $[p^* - 0.08, p^* + 0.08]$ sul validation set, su 10 seed indipendenti ho ottenuto una media di **23.1 FP contro 23.6 FN**, riducendo lo scarto medio a soli **0.5 campioni** su 306 di test.
Inoltre, nei rari casi di disaccordo tra le due reti ($M_1 \neq M_2$), un meta-modello logistico calibrato sui vettori latenti interviene per dirimere i campioni alla frontiera ambigua.

Una cosa interessante da notare è che giostrando la penalità dei falsi negativi e la penalità dei falsi positivi si possono ottenere risultati diversi in termini di accuratezza e F1 score. Di fatto si può addestrare una rete con uno sbilanciamento FP-FN desiderato con un'accuratezza alta. 

### Metriche della miglior run
```
==================================================
              REPORT DEL RECORD
==================================================
  Test Accuracy        :  87.91%
  F1-Score             :  0.889
  Recall (Vini Buoni)  :  90.85%
  Specificità          :  84.51%
--------------------------------------------------
  Matrice di Confusione:
                 Pred: Bad    Pred: Good
    Reale: Bad       120          22   (FP)
    Reale: Good       15         149   (TP)
==================================================
```

---

## 7. Tabellone di Tutti gli Esperimenti

Riepilogo delle tappe sperimentali con relativi risultati e considerazioni:

| # | Esperimento / Setup | Best Test Acc | Media Test | Gap Overfitting | Considerazioni |
|---|---------------------|:-------------:|:----------:|:---------------:|----------------|
| 1 | **Baseline 11 $\to$ 2** (Cross-Entropy standard) | ~75.00% | ~74.50% | ~8.0% | Collo di bottiglia troppo stretto, nessun ordine tra classi |
| 2 | **Rete Singola 88 $\to$ 10** + Cost-Sensitive Loss | 77.52% | 76.50% | 6.5% | Impostazione corretta, ma limitata dal covariate shift |
| 3 | **MoE v1** (Entropy Gating A/B) | 80.13% | 78.50% | ~4.5% | Migliora i FP ma aggiunge rigidità |
| 4 | **MoE v2** (Feature Gating con 12 feature) | 78.50% | 77.00% | 5.5% | Forte overfitting della rete di gating |
| 5 | **Split K-Means** (Clustering nello spazio X) | 76.80% | ~75.50% | 6.0% | I cluster sferici non mappano la densità reale |
| 6 | **Kennard-Stone + 2 Reti NCL** | 87.58% | 85.49% | 2.05% | Salto netto: interpolazione pura nello spazio |
| 7 | **Cascata Ortogonale + OOD Gating** | 80.07% | 79.02% | **0.16%** | Overfitting azzerato ma vincoli troppo rigidi |
| 8 | **Unificato Finale (KS + NCL + Ortogonalità + Bayes)** | **87.91%** | **84.74%** | **2.47%** | **Miglior compromesso: record e bilanciamento FP/FN** |

---

## 8. Struttura del Codice e Riproducibilità

Il codice è organizzato in moduli indipendenti e riproducibili:

```text
├── src/
│   ├── config/
│   │   └── config.py               # Parametri globali, loss weights e soglie Bayesiane
│   ├── preprocessing/
│   │   ├── remove_outliers.py      # Filtraggio dei valori chimicamente non plausibili
│   │   ├── adding_label_var.py     # Creazione label ordinale e dei 4 indici derivati
│   │   ├── splitting.py            # Campionamento Kennard-Stone e Adversarial Validation
│   │   └── standardization.py      # Standardizzazione (StandardScaler) coerente train/val/test
│   ├── model/
│   │   └── model.py                # WineQualityClassifier (12 -> 88 -> 10 -> 10) con estrazione latente
│   ├── training/
│   │   ├── loss.py                 # Hybrid Loss: CostSensitive + NCL Covariance + Ortogonalità
│   │   └── training.py             # Training loop congiunto con Cosine Annealing
│   ├── evaluation/
│   │   └── evaluation.py           # Calibrazione Bayesiana dinamica e Disagreement Gating
│   ├── data_analysis/
│   │   ├── raw_data_description.py # Statistiche descrittive e grafici pre-processing
│   │   └── processed_data_description.py # Analisi dati elaborati post-scaling
│   └── main.py                     # CLI modulare per eseguire singoli step
├── run_experiments.py              # Runner per benchmark su 10 seed statistici
├── requirements.txt                # Dipendenze con supporto PyTorch CUDA
└── README.md                       # Sei qui!
```

### Eseguire la pipeline completa da CLI
Per eseguire i singoli passaggi tramite la CLI in `src/main.py`:
```powershell
python src/main.py --action describe_raw_data
python src/main.py --action remove_outliers
python src/main.py --action add_label_var
python src/main.py --action split
python src/main.py --action standardize
python src/main.py --action describe_processed_data
python src/main.py --action train
python src/main.py --action evaluate
```

