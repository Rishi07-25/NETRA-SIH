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

## 8. Known Limitations & Handoff to Stage 3B
- **Stage 3A Scope:** Stage 3A establishes the data contract only. It contains no neural weights or forecasting inference.
- **Real Dataset Availability:** When CIC-IDS2017 is absent locally, the pipeline executes against deterministic fixtures.
- **Stage 3B Roadmap:** Stage 3B will consume the $(N, L, D)$ tensors and future targets generated by this module to train LSTM/GRU world models and evaluate early-warning risk calibration.
