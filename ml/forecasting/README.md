# Stage 3A: Real Dataset & Temporal Forecasting Foundation

## Overview
`ml/forecasting/` establishes the formal data contract and temporal sequence engineering pipeline required by the future Stage 3B temporal world model (e.g. LSTM / Transformer).

```text
Flow Telemetry (CIC-IDS2017)
      ↓
Normalized Flow Features (dataset_loader.py, schema.py)
      ↓
Temporal Network State S_t (temporal_state.py)
      ↓
State Sequence [S_(t-L+1), ..., S_t] (sequence_builder.py)
      ↓
Future Target at t + H (targets.py)
      ↓
Strict Partitioning & Preprocessing Isolation (temporal_split.py)
      ↓
Stage 3B Temporal World Model (Future Stage)
```

> **IMPORTANT:** Stage 3A implements **data engineering, schema contracts, temporal state aggregation, sliding sequence generation, future target labeling, and validation**. It does **NOT** implement neural forecasting models (no LSTM, Transformer, GNN, PyTorch, TensorFlow, or neural rollout). No forecasting performance claims are made at Stage 3A.

---

## 1. Primary Dataset Selection: CIC-IDS2017

### Rationale for Selection
1. **CICFlowMeter Feature Alignment:** CIC-IDS2017 features directly correspond to NETRA's flow telemetry schema (durations, packet/byte counters, rates, TCP flag distributions).
2. **Temporal Multi-Day Structure:** Traffic is captured across five distinct business days (Monday through Friday, July 3–7, 2017), providing realistic chronological progression.
3. **Distributed Attack Scenarios:** Different attack campaigns occur on specific days:
   - **Monday:** Benign baseline traffic only.
   - **Tuesday:** B-Profile FTP-Patator, SSH-Patator (Brute Force).
   - **Wednesday:** DoS Slowloris, SlowHTTPTest, Hulk, GoldenEye, Heartbleed.
   - **Thursday:** Web Attacks (Brute Force, XSS, SQL Injection), Infiltration.
   - **Friday:** Botnet, PortScan, DDoS.
4. **Natural Split Boundaries:** The multi-day structure permits leak-free day-based partitioning (e.g., Train: Mon–Wed, Val: Thu, Test: Fri) in addition to strict chronological splitting.

### Expected Dataset Structure
- **Local Path:** `data/raw/CIC-IDS2017/` (or user-specified directory/CSV file).
- Large dataset files are **git-ignored** (`data/raw/*`). Lightweight deterministic synthetic fixtures are used for continuous integration testing.

---

## 2. Canonical Schema Mapping (`schema.py`)

The schema provides deterministic mappings from raw dataset column aliases to canonical NETRA flow features:

| Canonical Flow Feature | CIC-IDS2017 Raw Column | UNSW-NB15 Raw Column | Description |
|:---|:---|:---|:---|
| `flow_duration` | `Flow Duration` | `dur` | Flow duration in microseconds |
| `tot_fwd_pkts` | `Total Fwd Packets` | `spkts` | Total forward packets |
| `tot_bwd_pkts` | `Total Backward Packets` | `dpkts` | Total backward packets |
| `tot_fwd_bytes` | `Total Length of Fwd Packets` | `sbytes` | Total forward payload bytes |
| `tot_bwd_bytes` | `Total Length of Bwd Packets` | `dbytes` | Total backward payload bytes |
| `flow_pkt_rate` | `Flow Packets/s` | (computed) | Packet transmission rate |
| `flow_byte_rate` | `Flow Bytes/s` | (computed) | Byte transmission rate |
| `syn_flag_cnt` | `SYN Flag Count` | (flags) | Count of SYN flags |
| `rst_flag_cnt` | `RST Flag Count` | (flags) | Count of RST flags |
| `ack_flag_cnt` | `ACK Flag Count` | (flags) | Count of ACK flags |
| `src_port` | `Source Port` | `sport` | Source transport port |
| `dst_port` | `Destination Port` | `dsport` | Destination transport port |
| `protocol` | `Protocol` | `proto` | IANA protocol (1=ICMP, 6=TCP, 17=UDP) |
| `timestamp` | `Timestamp` | `stime` | Event timestamp |
| `label` | `Label` | `attack_cat` | Attack or benign label |

---

## 3. Label Normalization & Operational Attack-Stage Taxonomy

### Canonical Attack Labels
Polymorphic raw labels are mapped to canonical categories:
- `BENIGN`, `Reconnaissance`, `Brute Force`, `DoS`, `DDoS`, `Web Attack`, `Infiltration`, `Botnet`, `Heartbleed`.

### Attack-Stage Taxonomy
To support strategic multi-step killchain analysis, NETRA maps attack classes to operational tactical stages:
- **Normal:** Benign background traffic.
- **Reconnaissance:** Port scans, host sweeps, IP sweeps.
- **Initial Access:** SSH-Patator, FTP-Patator, credential brute forcing.
- **Exploitation:** Web attacks (SQLi, XSS), Heartbleed.
- **Lateral Movement:** Botnet communication, post-exploitation infiltration.
- **Impact:** High-volume DoS / DDoS flooding attacks.

> [!IMPORTANT]
> **Taxonomy Disclaimer:** Attack stage is a NETRA-derived operational taxonomy, not a ground-truth label directly provided by the dataset.

---

## 4. Temporal Network State $S_t$ (`temporal_state.py`)

A state $S_t$ represents the aggregate behavioral snapshot of network traffic over a half-open window $[t, t + \Delta)$. It is **not** a single flow.

### Windowing Configuration
- **Window Size ($\Delta$):** Default 60 seconds. Captures macroscopic connection patterns.
- **Stride:** Default 30 seconds (50% window overlap for smooth temporal dynamics).

### 18 Predictive Features ($D = 18$)
1. `flow_count`: Total flows initiated in window.
2. `tot_fwd_pkts`: Total forward packets.
3. `tot_bwd_pkts`: Total backward packets.
4. `tot_fwd_bytes`: Total forward payload volume.
5. `tot_bwd_bytes`: Total backward payload volume.
6. `avg_pkt_size`: Mean packet size across all window flows.
7. `flow_duration_mean`: Mean flow duration.
8. `flow_pkt_rate_mean`: Mean packet transfer rate.
9. `flow_byte_rate_mean`: Mean byte transfer rate.
10. `syn_flag_cnt_tot`: Aggregate count of SYN flags.
11. `rst_flag_cnt_tot`: Aggregate count of RST flags.
12. `ack_flag_cnt_tot`: Aggregate count of ACK flags.
13. `fwd_bwd_pkt_ratio`: Ratio of forward to backward packets.
14. `fwd_bwd_byte_ratio`: Ratio of forward to backward bytes.
15. `unique_src_ports`: Cardinality of unique active client ports.
16. `unique_dst_ports`: Cardinality of unique contacted service ports.
17. `dst_port_entropy`: Shannon entropy of destination port distribution.
18. `tcp_ratio`: Fraction of TCP flows.
19. `udp_ratio`: Fraction of UDP flows.

### Target & Metadata Separation
The following fields are strictly quarantined as **metadata / targets** and NEVER included in $S_t$:
- `dominant_label`, `has_attack`, `attack_flow_ratio`, `attack_stage`, `window_id`, `window_start`, `window_end`.

---

## 5. Sequence Construction & Future Targets

### Sliding Sequence Tensor $X_t$ (`sequence_builder.py`)
- **Sequence Length ($L$):** Default 5 windows (2.5 minutes of observation).
- **Forecast Horizon ($H$):** Default 1 window ahead.
- **Tensor Shape:** $(N, L, D)$ where $N$ is sequence count, $L$ is lookback steps, $D$ is feature dimension.

### Future Target Vector $y_{t+H}$ (`targets.py`)
Derived **strictly** from the target window $t + H$:
1. `future_attack_binary`: $\{0, 1\}$ indicator whether any attack occurs at $t + H$.
2. `future_attack_type`: Dominant canonical attack label in window $t + H$.
3. `future_attack_stage`: NETRA-derived operational stage at $t + H$.
4. `future_attack_risk_score`: Fraction of attack flows in window $t + H \in [0.0, 1.0]$.

> [!CAUTION]
> **Strict Temporal Ordering:** $t_{\text{obs\_end}} \le t_{\text{target\_start}}$. No target information leaks into $X_t$.

---

## 6. Partitioning & Leakage Prevention (`temporal_split.py`)

### Split Strategies
1. **Chronological Splitting:** 60% Train, 20% Validation, 20% Test ordered by timestamp.
2. **Day-Based Scenario Splitting:** Partitions entire days (e.g. Mon–Wed Train, Thu Val, Fri Test).

### Key Leakage Protections
- **Partition-First Sequence Building:** Sequences are generated **independently** within each temporal partition. No sequence or target window ever crosses a partition boundary.
- **Preprocessing Isolation:** `StandardScaler` is fitted **strictly on training sequences**, then used to transform validation and test tensors.
- **Identifier Masking:** IP addresses, session IDs, and flow IDs are stripped from predictive inputs to prevent memorization.

---

## 7. Execution & Dataset Validation (`validate_dataset.py`)

Run validation on any CSV or directory of CSVs:
```bash
# Validate sample synthetic flows
python -m ml.forecasting.validate_dataset --input data/samples/synthetic_network_flows.csv --output_dir data/temporal

# Validate real CIC-IDS2017 dataset (when available locally)
python -m ml.forecasting.validate_dataset --input data/raw/CIC-IDS2017/ --format CIC-IDS2017 --output_dir data/temporal
```

Generated artifacts:
- `data/temporal/temporal_states.csv`: Aggregated states $S_t$.
- `data/temporal/train_sequence_metadata.csv`: Window timestamps for each sequence.
- `data/temporal/train_forecast_targets.csv`: Future targets at $t + H$.
- `data/temporal/stage3a_validation_report.json`: Comprehensive data quality summary.

---

---

## 8. Stage 3B.1: Temporal Forecasting Foundation (`world_model.py`)

Stage 3B.1 implements NETRA's neural temporal forecasting baseline using a 2-layer LSTM encoder and four specialized prediction heads.

### Model Specification
- **Input Dimension:** $D = 19$ behavioral network features.
- **Temporal Window Lookback:** $L = 5$ states ($X \in \mathbb{R}^{N \times 5 \times 19}$).
- **LSTM Encoder:** `input_size=19`, `hidden_size=64`, `num_layers=2`, `dropout=0.2`, `batch_first=True`.
- **State Embedding:** Final temporal hidden vector $h_t \in \mathbb{R}^{64}$.

---

## 9. Stage 3B.2: Full Temporal World Model & Autoregressive State Rollout

Stage 3B.2 upgrades NETRA into a genuine temporal **World Model** that predicts both future threat indicators and the continuous evolution of network behavioral state, supporting multi-step autoregressive rollout and horizon degradation analysis.

### Conceptual Architecture
```text
S_t (Observed Sequence: [S_(t-4), ..., S_t])
  ↓
Temporal Encoder (2-layer stacked LSTM, hidden_size=64)
  ↓
Latent State Embedding h_t in R^64
  ├─────────────────────────────────────────────────┐
  ▼                                                 ▼
Threat Forecasting Heads                    Next-State Prediction Head
- future_attack_binary (2 logits)           - predicted_state_(t+1) in R^19
- future_attack_type (9 logits)               (Standardized continuous state)
- future_attack_stage (6 logits)
- future_attack_risk_score (1 Sigmoid in [0,1])
  │                                                 │
  │                                                 ▼
  │                                         Recursive Autoregressive Shift:
  │                                         [S_(t-3), ..., S_t, S_hat_(t+1)]
  │                                                 ↓
  │                                         predict S_hat_(t+2) + Threats_(t+2)
  │                                                 ↓
  │                                         ... up to horizon K
  ▼                                                 ▼
Single-Step Threat Forecast                 Multi-Horizon Trajectory Simulation
```

### Five Prediction Heads
1. **Head 1 — Binary Attack Classifier (`future_attack_binary`):**
   - Output: 2 logits (Benign vs Attack)
   - Loss: `CrossEntropyLoss`
   - Confidence: `future_attack_probability` derived via Softmax ($P(\text{Attack})$)
2. **Head 2 — Attack Type Classifier (`future_attack_type`):**
   - Output: 9 logits corresponding to canonical categories
   - Loss: `CrossEntropyLoss` (conditionally masked for benign targets)
3. **Head 3 — Operational Attack Stage Classifier (`future_attack_stage`):**
   - Output: 6 logits corresponding to tactical progression stages
   - Loss: `CrossEntropyLoss` (conditionally masked for benign targets)
   - Disclaimer: Operational taxonomy, not ground truth.
4. **Head 4 — Attack Risk Score Regressor (`future_attack_risk_score`):**
   - Output: Sigmoid continuous value $\in [0.0, 1.0]$
   - Loss: `MSELoss`
   - Semantics: Empirical fraction of attack flows in future window (not a calibrated probability).
5. **Head 5 — Next-State Regressor (`head_state`):**
   - Architecture: `Linear(64, 64) -> ReLU -> Dropout(0.2) -> Linear(64, 19)`
   - Output: Continuous standardized state vector $\hat{S}_{t+1} \in \mathbb{R}^{19}$ (No Sigmoid applied)
   - Loss: `SmoothL1Loss`

### Multi-Task Loss Formulation
$$L_{\text{total}} = w_{\text{state}} L_{\text{state}} + w_{\text{bin}} L_{\text{binary}} + w_{\text{type}} L_{\text{type}} + w_{\text{stage}} L_{\text{stage}} + w_{\text{risk}} L_{\text{risk}}$$
Default weights:
- $w_{\text{state}} = 1.0$
- $w_{\text{bin}} = 1.0$
- $w_{\text{type}} = 1.0$
- $w_{\text{stage}} = 1.0$
- $w_{\text{risk}} = 2.0$ (Weighted $2.0\times$ because MSE on $[0, 1]$-bounded risk score produces smaller gradient scales compared to multi-class CrossEntropy).

### Benign Attribution Masking (`mask_benign_attribution=True`)
- When enabled, $L_{\text{type}}$ and $L_{\text{stage}}$ are computed strictly over attack samples (`binary_target == 1`), eliminating arbitrary classification penalty on benign background traffic.
- Binary, risk, and state losses are computed across all samples.

### Autoregressive State Rollout (`rollout.py`)
- **Engine:** `StateRolloutEngine` executes recursive multi-step simulation up to horizon $K \ge 1$.
- **Algorithm:** At step $k$, predicts $\hat{S}_{t+k}$, shifts the sequence by appending $\hat{S}_{t+k}$ and dropping the oldest state, then predicts step $k+1$ without teacher forcing or future ground-truth leakage.
- **Safety Checks:** Enforces $K \ge 1$, input shape $(N, 5, 19)$, finite values, `model.eval()`, `torch.no_grad()`.

### Naive Persistence Baseline
- Defines naive stationary forecast: $\hat{S}_{t+k} = S_t$.
- Evaluates LSTM state rollout against persistence across horizons $K \in \{1, 2, 3, 5\}$ to prove neural forecasting value over static assumptions.

### Horizon Degradation Analysis
- Measures State MAE/RMSE and Threat F1/MAE over increasing forecast horizons $K \in \{1, 2, 3, 5\}$.
- Produces degradation tables to observe error progression across simulation depth.

### Checkpoint v2 Bundle (`temporal_world_model_v2.pt`)
Contains:
- Model architecture (`input_size`, `hidden_size`, `num_layers`, `dropout`)
- State dict (including `head_state`)
- Label encodings (`type_to_idx`, `stage_to_idx`)
- Feature names (19 canonical predictive features)
- Fitted `StandardScaler`
- Training configuration & loss weights
- Validation & test metrics

### Execution & CLI
```bash
# Train Stage 3B.2 world model v2
python -m ml.forecasting.train_world_model --input data/samples/synthetic_network_flows.csv --epochs 25 --save_dir ml/models

# Using Real CIC-IDS2017 (when local directory is present)
python -m ml.forecasting.train_world_model --input data/raw/CIC-IDS2017/ --epochs 50 --save_dir ml/models
```

### Synthetic Fixture Limitation
> **WARNING:** Synthetic fixture result — not representative of CIC-IDS2017 benchmark performance. Small synthetic fixtures validate pipeline contracts and tensor shapes only.

