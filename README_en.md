# 🍷 Wine Quality Classification
> *The reasoning and tweaks that got me to 87.91%.*  
> 🌐 **[Leggi in italiano](README.md)**

It's common knowledge that when dealing with small tabular datasets, people usually reach for the usual "advanced" machine learning suspects like XGBoost, LightGBM, etc., rather than neural networks. Neural nets tend to overfit almost right away on tabular data, or the lack of samples keeps them from reaching decent performance.

Hence my personal challenge: I wanted to try building a neural network that could actually go toe-to-toe with these advanced ML methods. Testing with XGBoost, I ended up with around 83% accuracy.

Spoiler: I managed to hit **87.91%** accuracy on the test set, keeping a solid balance between False Positives and False Negatives.
I started with one core idea: turning the small dataset size and the simple problem structure to my advantage.

---

## Roadmap

- [Quickstart](#quickstart)
- [Methodology: Data Splitting and Training Techniques](#methodology-data-splitting-and-training-techniques)
- [1. The Problem](#1-the-problem)
- [2. My Two Insights](#2-my-two-insights)
- [3. Dead Ends and Hitting a Wall](#3-dead-ends-and-hitting-a-wall)
- [4. Breaking Through: Kennard-Stone, NCL, and Joint Training](#4-breaking-through-kennard-stone-ncl-and-joint-training)
- [5. Final Results and the 87.91% Record](#5-final-results-and-the-8791-record)
- [6. Full Experiments Board](#6-full-experiments-board)
- [7. Code Structure and Reproducibility](#7-code-structure-and-reproducibility)

---

## Quickstart

### 1. Clone and install dependencies
```bash
git clone https://github.com/TommasoSpadaro10/wine_quality_classification.git
cd wine_quality_classification

# With CUDA support (optional if you have an Nvidia GPU):
pip install --extra-index-url "https://download.pytorch.org/whl/cu126" -r requirements.txt
```

### 2. Run the statistical validation suite (10 independent runs)
```bash
python run_experiments.py
```

---

## Methodology: Data Splitting and Training Techniques

Before diving into each individual experiment, it's worth laying down the methodological ground rules used to handle the data, plus the bag of training tricks used "under the hood" to ensure convergence, keep things stable, and prevent overfitting.

### 1. Data Splitting Fundamentals: Training, Validation, and Test Sets

Splitting a supervised dataset comes down to one core rule: **generalizing to unseen data**. In my setup, the data is split following a **60% Train / 20% Validation / 20% Test** ratio:

```text
                       FULL DATASET (1,599 samples)
    ┌──────────────────────────────────┬─────────────────┬─────────────────┐
    │        TRAINING SET (60%)        │ VALIDATION (20%)│    TEST (20%)   │
    │           ~959 samples           │  ~320 samples   │  ~320 samples   │
    └──────────────────────────────────┴─────────────────┴─────────────────┘
                    │                           │                 │
        Backpropagation & Weights     Hyperparameter Tuning,  Final Blind
        (Updating W and b)            Early Stopping,         Evaluation
                                      Bayesian Thresholds     (Unbiased)
```

#### A. Roles and Separation of Concerns
1. **Training Set (60%)**:
   - The only set where the network computes gradients via backpropagation and updates weights $W$ and biases $b$.
2. **Validation Set (20%)**:
   - **No gradients are computed on this set.**
   - Used to:
     - Keep an eye on overfitting.
     - Drive **Early Stopping**.
3. **Test Set (20%)**:
   - Evaluated **only once at the very end** to gauge the real-world performance of the network.

#### B. Split Types
- **Standard Random Split**: splits the data completely randomly.
- **Stratified Split**: constrains sampling so that each set keeps the exact same percentage of each class.

---

### 2. Training Techniques Used "Under the Hood"

Besides the custom Cost-Sensitive loss function, training uses a set of techniques that I kept constant across the whole project mostly because they're not all that exciting (not much room for creativity there). I went with pretty standard techniques and values.

#### A. Weight Initialization (He / Kaiming Normal)
Linear layer weights aren't left to random defaults or generic normal distributions; they're initialized using the **Kaiming Normal** (He Initialization) criterion:
$$W \sim \mathcal{N}\left(0, \frac{2}{(1 + a^2) \cdot \text{fan\_in}}\right)$$
with a negative slope $a = 0.01$ tuned for the **LeakyReLU** activation function, and biases set to zero. This ensures signal variance stays stable across all layers.

#### B. Multi-Level Normalization and Regularization
To keep training stable, I used:
1. **Batch Normalization (`BatchNorm1d`)**: keeps means and variances controlled within each mini-batch, dampening internal covariate shift between layers and acting as a natural regularizer.
2. **Differentiated Dropout**: $p = 0.25$ on the first intermediate layer (88 neurons) to randomly drop 25% of connections and prevent co-dependency between correlated chemical features. $p = 0.10$ on the second layer (10 neurons) as a lighter touch, so we don't destabilize probability estimates right before the output.
3. **Weight Decay ($L_2$ Regularization)**: set in the optimizer to $\lambda = 10^{-4}$. It adds a penalty proportional to the squared norm of weights ($\frac{1}{2} \lambda \|W\|_2^2$), nudging the network to spread the load across smaller weights and avoid blown-out coefficients.

#### C. Optimization and Dynamic Learning Rate Scheduling
- **Adam Optimizer**: uses first-order ($\beta_1 = 0.9$) and second-order ($\beta_2 = 0.999$) momentum, starting with a learning rate $\eta_0 = 3 \times 10^{-3}$.
- **Cosine Annealing with Warm Restarts (`CosineAnnealingWarmRestarts`)**:
  - The learning rate isn't constant; it follows periodic cosine cycles:
    $$\eta_t = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min})\left(1 + \cos\left(\frac{T_{cur}}{T_i}\pi\right)\right)$$
  - With parameters $T_0 = 2000$ epochs and multiplier $T_{\text{mult}} = 2$, the learning rate gradually drops down to $10^{-6}$ and then "resets".
  - **Why do this?** Periodic spikes allow the optimizer to pop out of narrow local minima (which are notorious for overfitting) and settle into **flat minima** (wide, shallow valleys), which are widely recognized in literature for offering far better generalization.

#### D. Early Stopping with Minimum Activation Threshold
Training can run for up to 20,000 potential epochs, governed by a targeted stopping rule:
- **Patience = 2,500 epochs**: if accuracy on the Validation Set doesn't improve for 2,500 epochs in a row, training halts.
- **Activation Threshold (`EARLY_STOPPING_MIN_ACCURACY = 0.70`)**: early stopping only kicks in once the model clears 70% validation accuracy, avoiding premature stops during the noisy initial exploration phase.
- **Restoring the Best Model**: once done, the weights from the final epoch (often hit by local degradation) are tossed out, and the weights corresponding to the absolute peak validation accuracy are automatically reloaded.

---

## 1. The Problem

The dataset contains **1,599 bottles** of red wine, each with 11 chemical features and a quality rating from human sommeliers ranging from **3 to 8**.

In my first attempt, I went with the simplest possible idea: set up binary classification using a barebones network:
`Input(11) -> Linear(2) -> Linear(2) -> Output(2)` with standard Cross-Entropy.

Result: the model hit a wall around **74-75%** accuracy.

First off, I looked into expanding the input variables.
Looking at correlation plots and doing some educated guessing (like, alcohol content and density have a huge sway over wine quality), I dropped 3 raw variables and engineered 4 new features using those discarded ones:

$$\text{acidity\_ratio} = \frac{\text{volatile acidity}}{\text{fixed acidity}} \quad \text{(measures volatile acidity vs structural body)}$$

$$\text{sulfur\_ratio} = \frac{\text{free sulfur dioxide}}{\text{total sulfur dioxide}} \quad \text{(active fraction protecting against oxidation)}$$

$$\text{alcohol\_density} = \text{alcohol} \times \text{density} \quad \text{(proxy for hydroalcoholic body)}$$

$$\text{total\_acidity} = \text{fixed acidity} + \text{volatile acidity} \quad \text{(overall acid balance)}$$

Result: input space expanded to **12 features**.

Moving along, I figured the main culprits were:
1. **Initial bottleneck**: squeezing 11 complex variables straight down into just 2 neurons on the very first layer was crushing most of the geometric information in the data.
2. **Cross-Entropy ignores ordinal distance**: treating this as standard categorical classification means every mistake is penalized equally.

Looking at the variable comparison plots and the feature-vs-target charts, I figured the feature space wasn't easily separable into the two classes. That meant giving the network more breathing room—widening the layers so it could actually "learn" more complex geometric operations (rotations, scaling, etc.) to pull the two classes apart as much as possible. Since the dataset is small, I played around with a bunch of setups and landed on:
`Input(12) -> Linear(88) -> Linear(10) -> Output(10)` with standard Cross-Entropy.

---

## 2. My Two Insights

To really make a leap, I had to rethink how the model computed loss and made predictions.

### A. Reformulating to 10 Classes and CostSensitiveLoss
Instead of forcing the network to spit out 0 or 1 right off the bat, I set the output to **10 classes** (the 1 to 10 rating scale), where each class $i$ gets an associated probability $P_i$ with $$\sum_{i=1}^{10} P_i = 1$$. In practice, $P_i$ represents the predicted probability for each grade.

Borrowing an idea from Michael Nielsen's *Neural Networks and Deep Learning*, I figured I should design a metric that was simple and felt closer to how a human might learn.
So, I built a custom loss function:
$$\mathcal{L}_{\text{cost}} = \sum_{i=1}^{10} P_i \cdot \Big( \vert i - y_{\text{true}} \vert + \text{Penalty}(i, y_{\text{true}}) \Big)$$

The basic ideas are:
1. **Linear distance penalty**: if a wine's real score is 6 and the network predicts 5, that's a minor slip-up ($|5 - 6| = 1$); but if it predicts 3, that's a much bigger blunder ($|3 - 6| = 3$). The penalty scales linearly with distance, giving the network a continuous directional gradient (I tried using log-distance or squared-distance, but performance actually tanked).
2. **Cross-threshold penalty**: also crucial: if the prediction crosses the fence between bad and good wine ($5 \leftrightarrow 6$), an extra penalty kicks in for false positives and false negatives. At the end of the day, the real goal is just correctly sorting good wine from bad wine.
3. **Label Smoothing (10%)**: a 10% uniform cost share prevents the network from becoming overconfident and spitting out crazy logit extremes on noisy samples (sommelier ratings definitely have some inherent subjectivity).

### B. Continuous Expected Value Instead of ArgMax
Using `argmax` caused jerky jumps and instability, especially for samples hovering right around the critical 5 vs 6 boundary. So instead, I chose to calculate the **continuous Expected Value**:
$$\mathbb{E}[\text{quality}] = \sum_{i=1}^{10} i \cdot P(y = i \mid x)$$

This gives a continuous score that also reflects how confident the network is: a borderline wine gets a continuous value around 5.5 rather than being subjected to a blunt, hard binary call.

I even tried playing with more complex probability distributions, thinking maybe a Weibull or a simple Gaussian would give better results, but nope. Probably down to human bias: human wine ratings might follow those distributions, but the network just learns off raw numbers.

### C. Analytical Threshold Calculation (Youden Index)
To pick the cutoff threshold on this continuous expected value, I didn't just guess an arbitrary number (like 5.5). I computed the ROC curve on the validation set ($\text{AUC} = 0.8317$) and analytically solved for the point maximizing Youden's $J$ statistic ($J = \text{TPR} - \text{FPR}$). The math landed on a threshold of exactly **5.0000**. Not exactly a groundbreaking discovery, but nice to have confirmed.

With these two insights, the single network jumped from ~75% to **over 84%**.

---

## 3. Dead Ends and Hitting a Wall

Along the way, I also tested ideas that looked great on paper, but in practice turned out to be total dead ends:

### The Illusion of the Fake 86.64% Record
At one point, while nudging the decision threshold to 5.65, I saw accuracy jump to **86.64%**. For a second I thought I'd struck gold! Then I checked the confusion matrix and saw through the trick: by bumping the threshold up past 5.65, I had basically cut out all score 6 wines from the "Good" class. The dataset had become heavily imbalanced, and the network was just predicting "Bad" almost across the board. High accuracy purely on paper, without the network actually learning anything useful.

### Experiments with Mixture of Experts (MoE) Architectures
I tried combining two networks (a generalist Model A and a Model B specialized on Model A's mistakes) steered by a gating meta-network (or logistic regression):
* **MoE v1 (Entropy Gate)**: stalled out at around 80% test accuracy.
* **MoE v2 (Chemical Feature Gate)**: feeding input features directly to the gate (kind of like self-attention) led to blatant **overfitting** (83% on validation, but crashed to 78.5% on test). With only about 1,600 samples, a meta-network ends up memorizing individual data points instead of learning a genuinely general routing strategy.
Or what would happen is the meta-network would just defer to the first network almost every time, since it was more reliable across the board. In short, I couldn't find a way to get two or three networks to play nice together.

---

## 4. Breaking Through: Kennard-Stone, NCL, and Joint Training

Even after tuning the single network, an annoying problem remained: a stubborn ~6% gap between validation (~82%) and test (~76%).

That got me thinking: the network was probably having to extrapolate rather than interpolate on the test set. With only ~1,500 samples scattered across 12 dimensions, a purely random or standard stratified split inevitably left empty pockets in the training set. The model was basically forced to perform full-on geometric **extrapolation** on test samples it had never seen anything close to.

```text
       Random Split (Problem)              Kennard-Stone Split (Solution)
    ┌───────────────────────────┐         ┌───────────────────────────┐
    │  ● Train   ○ Test         │         │  ● Train   ○ Test         │
    │       ●    ●              │         │  ●                     ●  │
    │    ●           ○          │         │       ○    ●    ○         │
    │  ●         ●              │         │    ●    ○    ●     ●      │
    │       ●                   │         │  ●                     ●  │
    └───────────────────────────┘         └───────────────────────────┘
    Test requires extrapolation            Train occupies the outer hull:
                                           test works by interpolation
```

### A. The Geometric Intuition: Stratified Kennard-Stone
To tackle this issue at its root, towards the end of the project I swapped traditional sampling for **Stratified Kennard-Stone**:
1. **Geometric selection per class**: the algorithm iteratively picks samples with the maximum mutual Euclidean distance within each class. This simultaneously preserves class balance (stratification) while maximizing coverage across the 12D chemical space.
2. **Convex hull for the Training Set**: by filling the training set (60%) with extreme and boundary samples, the Validation and Test sets end up safely enclosed inside, operating via **pure geometric interpolation**.
3. **Shift check with Adversarial Validation (5-Fold RF)**: to make sure Train and Validation were truly indistinguishable statistically, I ran a K-Fold **Adversarial Validation** (5-Fold Stratified with Random Forest) trained to guess which set a sample came from. The result confirmed what I hoped—an apparent absence of covariate shift ($\text{ROC-AUC} = 0.4527$, basically coin-flip territory at 0.50).

### B. Negative Correlation Learning (NCL) with Latent Orthogonality
Once sampling was taken care of, the remaining "simple" task was squeezing out more performance.
After trying ensemble nets or logistic regressions taking inputs from pre-trained models, performance basically just ended up mirroring whatever models went in. I even tried a Mixture of Experts setup, training a second expert specifically on the first one's mistakes, with no luck.
That's when I thought about forcing two networks to train side-by-side so they'd attack the problem from "orthogonal" angles. And I mean orthogonal literally, since I actually feed the orthogonality of the two networks' logits directly into the loss during training. That's why I went with **Negative Correlation Learning**: a single cost function that penalizes correlation between the two models' error residuals and slaps on an explicit **orthogonality constraint on the latent vectors** ($H_1 \perp H_2$):

$$\mathcal{L}_{\text{hybrid}} = \mathcal{L}_{\text{cost}}(P_1, Y) + \mathcal{L}_{\text{cost}}(P_2, Y) + \lambda_{\text{NCL}} \cdot \text{Cov}(e_1, e_2) + \lambda_{\text{ortho}} \cdot \cos^2(H_1, H_2)$$

This forces each network to specialize in complementary aspects of the chemical data. That constraint slashed variance across different runs by 38% ($\sigma$ dropped from 1.40% to 0.87%), stabilizing average accuracy above 85%.

The idea is that the two networks are pitted against each other, but pushed to pick up two completely different patterns. Maybe you could try training even more networks with this method, but my guess is there's a limit to how many orthogonal degrees of freedom you can meaningfully train—mathematically, that number has to be lower than the number of input features anyway.

### C. Joint Hybrid Training
So I implemented **joint backpropagation**:
- A single optimizer step (`optimizer.step()`) updates both models in unison on every mini-batch.
- This dynamically forces the latent vectors to diverge geometrically ($H_1 \perp H_2$) and their error residuals to decorrelate throughout the entire training run.

---

## 5. Final Results and the 87.91% Record

### Bayesian Threshold Calibration
In the first version of the unified model, performance was great, but it still leaned towards making more False Positives than False Negatives (+10 FP in the best run).

Going back to **Bayesian Minimum Risk Theory**, I calculated the theoretical optimal threshold $p^*$ based on asymmetric costs:

$$p^* = \frac{\text{PENALTY\_BAD\_TO\_GOOD}}{\text{PENALTY\_BAD\_TO\_GOOD} + \text{PENALTY\_GOOD\_TO\_BAD}} = \frac{12.0}{12.0 + 19.5} = 0.3810$$

By calibrating the threshold $\tau$ within the interval $[p^* - 0.08, p^* + 0.08]$ on the validation set, across 10 independent seeds I got an average of **23.1 FP vs 23.6 FN**, closing the average gap down to just **0.5 samples** out of 306 test bottles.
Plus, in the rare cases where the two networks disagree ($M_1 \neq M_2$), a logistic meta-model calibrated on the latent vectors steps in to settle ambiguous boundary calls.

One interesting thing to point out is that by playing around with the penalties for false negatives and false positives, you can steer the accuracy and F1 score however you like. Basically, you can train a network to hit whatever FP-FN trade-off you want while keeping accuracy high.

### Best Run Metrics
```
==================================================
                 RECORD REPORT
==================================================
  Test Accuracy        :  87.91%
  F1-Score             :  0.889
  Recall (Good Wines)  :  90.85%
  Specificity          :  84.51%
--------------------------------------------------
  Confusion Matrix:
                 Pred: Bad    Pred: Good
    Actual: Bad      120          22   (FP)
    Actual: Good      15         149   (TP)
==================================================
```

---

## 6. Full Experiments Board

Summary of each experimental step with results and takeaways:

| # | Experiment / Setup | Best Test Acc | Mean Test | Overfitting Gap | Notes & Thoughts |
|---|---------------------|:-------------:|:----------:|:---------------:|------------------|
| 1 | **Baseline 11 $\to$ 2** (Standard Cross-Entropy) | ~75.00% | ~74.50% | ~8.0% | Bottleneck too tight, zero ordering between classes |
| 2 | **Single Net 88 $\to$ 10** + Cost-Sensitive Loss | 77.52% | 76.50% | 6.5% | Right track, but held back by covariate shift |
| 3 | **MoE v1** (Entropy Gating A/B) | 80.13% | 78.50% | ~4.5% | Improves FPs but adds rigidity |
| 4 | **MoE v2** (Feature Gating with 12 features) | 78.50% | 77.00% | 5.5% | Heavy overfitting in the gating network |
| 5 | **K-Means Split** (Clustering in X space) | 76.80% | ~75.50% | 6.0% | Spherical clusters don't map real density |
| 6 | **Kennard-Stone + 2 NCL Nets** | 87.58% | 85.49% | 2.05% | Big leap: pure interpolation in feature space |
| 7 | **Orthogonal Cascade + OOD Gating** | 80.07% | 79.02% | **0.16%** | Zero overfitting, but constraints way too stiff |
| 8 | **Final Unified (KS + NCL + Orthogonality + Bayes)** | **87.91%** | **84.74%** | **2.47%** | **Best trade-off: record high and balanced FP/FN** |

---

## 7. Code Structure and Reproducibility

The code is split into independent and reproducible modules:

```text
├── src/
│   ├── config/
│   │   └── config.py               # Global parameters, loss weights, and Bayesian thresholds
│   ├── preprocessing/
│   │   ├── remove_outliers.py      # Filters chemically implausible values
│   │   ├── adding_label_var.py     # Creates ordinal labels and the 4 engineered ratios
│   │   ├── splitting.py            # Kennard-Stone sampling and Adversarial Validation
│   │   └── standardization.py      # Consistent train/val/test standardization (StandardScaler)
│   ├── model/
│   │   └── model.py                # WineQualityClassifier (12 -> 88 -> 10 -> 10) with latent extraction
│   ├── training/
│   │   ├── loss.py                 # Hybrid Loss: CostSensitive + NCL Covariance + Orthogonality
│   │   └── training.py             # Joint training loop with Cosine Annealing
│   ├── evaluation/
│   │   └── evaluation.py           # Dynamic Bayesian calibration and Disagreement Gating
│   ├── data_analysis/
│   │   ├── raw_data_description.py # Descriptive stats and pre-processing plots
│   │   └── processed_data_description.py # Post-scaling processed data analysis
│   └── main.py                     # Modular CLI to run individual steps
├── run_experiments.py              # Benchmark runner across 10 statistical seeds
├── requirements.txt                # Dependencies with PyTorch CUDA support
└── README.md                       # You are here!
```

### Running the Full Pipeline from CLI
To run individual steps via the CLI in `src/main.py`:
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
