# Dataset Documentation

## Primary Benchmark Dataset: CIC-IDS2017

### Rationale for Selection in Stage 3
For NETRA's Stage 3 Temporal Attack Forecasting, **CIC-IDS2017** (Canadian Institute for Cybersecurity) has been selected as the primary benchmark dataset due to three foundational architectural properties:

1. **Feature Alignment with Flow Telemetry:** CICFlowMeter-extracted attributes (flow durations, packet rates, byte rates, inter-arrival times, TCP flags) directly match NETRA's flow data contracts.
2. **Multi-Day Temporal Structure:** Traffic spans five continuous workdays (Monday, July 3 to Friday, July 7, 2017), preserving realistic time progressions and natural scenario boundaries.
3. **Attack Progression & Scenario Isolation:**
   - **Monday:** Benign traffic exclusively (used for baseline calibration).
   - **Tuesday:** Brute Force (FTP-Patator, SSH-Patator).
   - **Wednesday:** DoS (Slowloris, SlowHTTPTest, Hulk, GoldenEye) and Heartbleed.
   - **Thursday:** Web Attacks (Brute Force, XSS, SQL Injection) and Infiltration.
   - **Friday:** Botnet, PortScan, and DDoS.

---

## Secondary Dataset Support: UNSW-NB15
`ml/forecasting/schema.py` provides cross-dataset alias mappings for UNSW-NB15 flow features (`dur`, `sbytes`, `dbytes`, `proto`, etc.) to facilitate future cross-dataset generalizability experiments without requiring UNSW-NB15 as the primary runtime dependency.

---

## Data Ingestion & Storage Policy
- **Storage Rules:** Large `.pcap`, `.pcapng`, and raw `.csv` files exceeding 50MB MUST NOT be checked into version control.
- **Directory Layout:**
  - `data/raw/CIC-IDS2017/`: Target local directory for raw CSV extracts.
  - `data/processed/`: Cleaned and normalized flow telemetry.
  - `data/features/`: Tabular Stage 2 feature matrices (`features_X.csv`, `labels_y.csv`).
  - `data/temporal/`: Stage 3A temporal forecasting states ($S_t$), sequence metadata, and targets.
  - `data/samples/`: Lightweight deterministic fixtures (`synthetic_network_flows.csv`) for testing.

---

## Canonical Flow Schema Mapping
Raw column headers are mapped to normalized NETRA features:
- `Flow Duration` $\rightarrow$ `flow_duration`
- `Total Fwd Packets` $\rightarrow$ `tot_fwd_pkts`
- `Total Backward Packets` $\rightarrow$ `tot_bwd_pkts`
- `Total Length of Fwd Packets` $\rightarrow$ `tot_fwd_bytes`
- `Total Length of Bwd Packets` $\rightarrow$ `tot_bwd_bytes`
- `Flow Packets/s` $\rightarrow$ `flow_pkt_rate`
- `Flow Bytes/s` $\rightarrow$ `flow_byte_rate`
- `SYN Flag Count` $\rightarrow$ `syn_flag_cnt`
- `RST Flag Count` $\rightarrow$ `rst_flag_cnt`
- `ACK Flag Count` $\rightarrow$ `ack_flag_cnt`
- `Destination Port` $\rightarrow$ `dst_port`
- `Protocol` $\rightarrow$ `protocol`
- `Timestamp` $\rightarrow$ `timestamp`
- `Label` $\rightarrow$ `label`

---

## NETRA Operational Attack-Stage Taxonomy
To analyze attack progression along a killchain, NETRA maps canonical attack classes into operational stages:
- **Normal:** BENIGN traffic
- **Reconnaissance:** PortScan, IPSweep
- **Initial Access:** SSH-Patator, FTP-Patator (Brute Force)
- **Exploitation:** Web Attacks, Heartbleed
- **Lateral Movement:** Botnet, Infiltration
- **Impact:** DoS, DDoS

> [!IMPORTANT]
> **Taxonomy Disclaimer:** Attack stage is a NETRA-derived operational taxonomy, not a ground-truth label directly provided by the dataset.
