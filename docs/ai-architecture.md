# AI & Machine Learning Architecture

## Methodology Overview
Network attack forecasting requires capturing both static flow signatures and longitudinal behavioral patterns. NETRA employs an ensemble multi-stage architecture:

```text
Time-Windowed Network Features: X(t - W : t)
                     │
     ┌───────────────┴───────────────┐
     ▼                               ▼
[Stage 1: Anomaly Detection]   [Stage 2: Attack Classification]
 Unsupervised Outlier Score     Multi-class Threat Attribution
     │                               │
     └───────────────┬───────────────┘
                     ▼
       [Stage 3: Risk Forecasting]
        Predicts: Risk(t + Δt)
        Horizon: 1min, 5min, 15min
                     │
                     ▼
       [Stage 4: Threat Score Engine]
        Normalized Risk Metric (0 - 100)
```

---

## 1. Feature Engineering & Time Windowing
- Aggregates raw packet flows into structured feature vectors over rolling windows ($W$).
- Metrics include: Packet arrival intervals, flow durations, byte ratios, TCP flag distributions, entropy of source/destination ports, and connection frequency per IP.

## 2. Stage 1: Behavioral Anomaly Detection
- **Purpose:** Detect deviations from standard operating baselines without prior attack signatures.
- **Approaches under evaluation:** Isolation Forest, One-Class SVM, or Deep Autoencoders.
- **Output:** Continuous anomaly anomaly score $\in [0, 1]$.

## 3. Stage 2: Attack Classification
- **Purpose:** Classify known threat categories (Reconnaissance, Port Scans, DoS, Brute Force).
- **Approaches under evaluation:** Gradient Boosted Trees (XGBoost / LightGBM) or Random Forest.
- **Output:** Probability distribution across threat classes.

## 4. Stage 3: Temporal Risk Forecasting
- **Purpose:** Model the progression of attack precursors over time to forecast future attack probability.
- **Approaches under evaluation:** Temporal Convolutional Networks (TCN), LSTM/GRU, or Autoregressive ML ensembles.
- **Output:** Forecasted attack risk $P(\text{Attack} \mid t + \Delta t)$.
