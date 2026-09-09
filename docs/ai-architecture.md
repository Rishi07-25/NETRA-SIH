# AI & Machine Learning Architecture

## Methodology Overview
Network attack forecasting requires capturing both immediate flow signatures and longitudinal behavioral patterns. NETRA employs a multi-stage progressive architecture:

```text
RAW NETWORK TRAFFIC / FLOW TELEMETRY (CIC-IDS2017)
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│ STAGE 1: DATA PIPELINE & FEATURE ENGINEERING            │
│ Flow Cleaning, Port/Protocol Features, Window Binning   │
└──────────────────────────┬──────────────────────────────┘
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
┌────────────────────────────────┐  ┌─────────────────────────────────────────┐
│ STAGE 2: CURRENT THREAT ENGINE │  │ STAGE 3A: TEMPORAL FORECAST FOUNDATION  │
│ - Random Forest Classifier     │  │ - Aggregate Network State S_t           │
│ - Isolation Forest Anomaly     │  │ - Sequence Tensors X_t in R^(N x L x D) │
│ - Logistic Regression Baseline │  │ - Future Targets at t + H               │
│ - Unified Threat Score         │  │ - Chronological / Day-Based Splitting   │
└────────────────────────────────┘  └────────────────────┬────────────────────┘
                                                         │
                                                         ▼
                                    ┌─────────────────────────────────────────┐
                                    │ STAGE 3B: TEMPORAL WORLD MODEL (FUTURE) │
                                    │ - Recurrent Neural Network (LSTM / GRU) │
                                    │ - Multi-Horizon Attack Risk Forecasting │
                                    │ - Attack Stage Killchain Progression    │
                                    └────────────────────┬────────────────────┘
                                                         │
                                                         ▼
                                    ┌─────────────────────────────────────────┐
                                    │ STAGE 4: BACKEND & REAL-TIME ENGINE     │
                                    │ - FastAPI Inference Service             │
                                    │ - Dynamic Risk Aggregator (0 - 100)     │
                                    │ - Early Warning Dashboard Telemetry     │
                                    └─────────────────────────────────────────┘
```

---

## 1. Stage 1: Data Pipeline & Feature Engineering
- Ingests raw packet captures and flow records.
- Standardizes timestamps, handles invalid/infinite metrics, and creates clean bidirectional flow records.
- Computes flow-level domain metrics: rates, durations, port entropy, protocol ratios, TCP flag distributions.

## 2. Stage 2: Current Threat Detection Engine
- **Purpose:** Evaluate the threat level of traffic observed right now ($X_t \rightarrow \hat{y}_t$).
- **Components:**
  - Supervised Random Forest multi-class attack classifier.
  - Unsupervised Isolation Forest behavioral anomaly detector.
  - Logistic Regression linear baseline.
  - Unified Threat Engine fusing classification confidence and anomaly telemetry into transparent threat levels (Low, Medium, High, Critical).

## 3. Stage 3A: Real Dataset + Temporal Forecasting Foundation
- **Purpose:** Establish the formal data contracts, temporal sequences, and target variables required for predictive attack forecasting without training neural networks prematurely.
- **Key Mechanics:**
  - **Temporal Network State ($S_t$):** Aggregate behavioral vector over half-open window $[t, t + \Delta)$ spanning 18 predictive features.
  - **Sliding Observation Window ($X_t$):** Sequence $[S_{t-L+1}, \dots, S_t] \in \mathbb{R}^{N \times L \times D}$ where $L=5$ windows (lookback).
  - **Future Target ($y_{t+H}$):** Target derived strictly from window $t+H$ (binary attack occurrence, canonical attack type, operational attack stage, attack flow ratio).
  - **Leakage Prevention:** Strict chronological and scenario-aware day partitioning. Sequences are built independently per partition. Preprocessing scalers are fit strictly on training partitions.
  - **Operational Attack-Stage Taxonomy:** Maps attack types into tactical stages (Normal, Reconnaissance, Initial Access, Exploitation, Lateral Movement, Impact).
  > **Taxonomy Disclaimer:** Attack stage is a NETRA-derived operational taxonomy, not a ground-truth label directly provided by the dataset.

## 4. Stage 3B: Temporal World Model & Autoregressive State Rollout
- **Stage 3B.1 (Temporal Forecasting Foundation):**
  - Sequence-to-threat neural network forecasting future attack dynamics at horizon $t + H$.
  - 2-layer LSTM temporal encoder (`input_size=19`, `hidden_size=64`, `dropout=0.2`) extracting a 64-dimensional latent network-state embedding.
  - Forecasting heads: `future_attack_binary`, `future_attack_type`, `future_attack_stage`, `future_attack_risk_score`.
- **Stage 3B.2 (World Model + Autoregressive Rollout):**
  - Predicts continuous standardized future network state $\hat{S}_{t+1} \in \mathbb{R}^{19}$ via `head_state` (`Linear(64, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 19)`).
  - Multi-task loss: $w_{\text{state}} L_{\text{state}} + w_{\text{bin}} L_{\text{binary}} + w_{\text{type}} L_{\text{type}} + w_{\text{stage}} L_{\text{stage}} + w_{\text{risk}} L_{\text{risk}}$ with configurable weights (defaults: state=1.0, binary=1.0, type=1.0, stage=1.0, risk=2.0).
  - Benign attribution masking: masks $L_{\text{type}}$ and $L_{\text{stage}}$ when `binary_target == 0` to eliminate semantic penalty on benign background flows.
  - Recursive state simulation: shifts predicted states $\hat{S}_{t+k}$ into the input sequence to project network evolution and threat indicators over horizons $K \in \{1, 2, 3, 5\}$ without ground-truth teacher forcing.
  - Persistence baseline: compares LSTM multi-step trajectory against static persistence baseline ($\hat{S}_{t+k} = S_t$) to validate predictive progression.
- **Distinction:** `future_attack_probability` reflects model confidence of an attack occurring, whereas `future_attack_risk_score` measures expected attack traffic volume density.
- **Limitation:** Tested on synthetic development fixtures. Real-world benchmark performance awaits full local CIC-IDS2017 ingestion.
