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

## 4. Stage 3B: Temporal World Model (Future Implementation)
- **Purpose:** Sequence-to-sequence neural model forecasting attack occurrence and stage transition ahead of execution.
- **Planned Models:** LSTM, GRU, or Temporal Convolutional Networks.
- **Target Horizons:** 1-step, 3-step, 5-step ahead forecasting.
