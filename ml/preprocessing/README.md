# Network Traffic Preprocessing Pipeline

## Overview
This package contains the reusable network traffic preprocessing pipeline for **NETRA (Network Threat Early-warning & Risk Analytics)**. It standardizes raw network flow logs, eliminates invalid values and duplicates, extracts tabular ML feature matrices, and produces chronological sliding time-windows for temporal risk forecasting.

---

## Architecture & Stages

```text
Raw Network Flow CSV (CIC-IDS2017 / UNSW-NB15 / Synthetic)
                          │
                          ▼
            [1. Data Loading & Inspection]
                          │
                          ▼
        [2. Header & Column Name Normalization]
              (snake_case, stripped characters)
                          │
                          ▼
             [3. Deduplication & Cleanup]
            (drop duplicate flow records)
                          │
                          ▼
      [4. Missing & Infinite Value Imputation]
       (infs -> NaN -> median/mean imputation)
                          │
                          ▼
            [5. Label Canonical Mapping]
      (BENIGN, Reconnaissance, Brute Force, DDoS)
                          │
            ┌─────────────┴─────────────┐
            ▼                           ▼
[6. Feature Matrix Extraction] [7. Sliding Window Generation]
   - Strict leakage prevention     - Chronological slicing
   - Drop IPs/IDs/Timestamps       - Window size: W (e.g. 60s)
   - Compute flow ratios           - Stride: S (e.g. 30s)
   - Outputs: X, y                 - Sequential risk features
```

---

## Module Breakdown

### 1. `clean_data.py`
- **`normalize_column_names(columns)`**: Normalizes column strings into clean `snake_case`.
- **`identify_label_column(df)`** & **`identify_timestamp_column(df)`**: Auto-detects ground truth and datetime fields dynamically.
- **`normalize_labels(label_series)`**: Maps heterogeneous attack labels across datasets (e.g., `PortScan`, `SSH-Patator`, `ddos-synflood`) into canonical classes (`BENIGN`, `Reconnaissance`, `Brute Force`, `DDoS`, `Exploitation`, `Botnet`).
- **`clean_network_dataframe(df, ...)`**: Replaces infinite floats with median values and eliminates exact duplicate connections.
- **`load_and_clean_dataset(filepath, ...)`**: High-level loader reading CSV files safely.

### 2. `feature_engineering.py`
- **`compute_derived_flow_features(df)`**: Generates packet ratios (`fwd_bwd_pkt_ratio`), byte ratios (`fwd_bwd_byte_ratio`), and average packet sizes (`avg_pkt_size`).
- **`prepare_feature_matrix(df, ...)`**: Partitions data into numeric feature matrix $X$ and target label series $y$. Strictly excludes identifier columns (`src_ip`, `dst_ip`, `flow_id`, timestamps) to avoid data leakage.

### 3. `create_time_windows.py`
- **`create_sliding_time_windows(df, window_size_sec, stride_sec, ...)`**: Sorts network flows chronologically and aggregates flow metrics within discrete temporal windows $[t_i, t_i + W)$. 
- Computes aggregate flow statistics, window flow counts, and attack ratios without leaking future timestamps into the current window.

### 4. `run_pipeline.py`
- CLI runner that accepts raw CSV input, runs all 3 pipeline stages, and exports:
  - Cleaned data: `data/processed/cleaned_flows.csv`
  - Feature matrix: `data/features/features_X.csv` and `data/features/labels_y.csv`
  - Temporal windows: `data/features/time_windows.csv`

---

## Dataset Compatibility

| Dataset | Expected Flow Schema | Notes |
|---|---|---|
| **CIC-IDS2017** | 80+ features (Flow Duration, Fwd/Bwd Packets, Flags) | Column names contain spaces; auto-normalized by `clean_data.py`. Contains infinite rates. |
| **UNSW-NB15** | 49 features (`dur`, `sbytes`, `dbytes`, `sttl`, `attack_cat`) | `sttl`/`stime` auto-detected as timestamp; `attack_cat` mapped to canonical labels. |
| **Synthetic Sample** | 17 representative flow fields | Used for local development and test validation (`data/samples/synthetic_network_flows.csv`). |

---

## How to Run the Pipeline

```bash
# Activate virtual environment
source .venv/bin/activate

# Execute pipeline on sample data
python ml/preprocessing/run_pipeline.py --input data/samples/synthetic_network_flows.csv --window-size 60 --stride 30
```

---

## Synthetic Sample Data vs Real Datasets

- **Synthetic Sample:** A lightweight 50-row CSV representing 4 minutes of sequential traffic across Benign browsing, Port Scanning, SSH Brute Force, and DDoS SYN flood. **It is NOT measured attack traffic** and is intended strictly for pipeline functional testing and local development without downloading multi-gigabyte files.
- **Real Benchmark Datasets:** Full PCAP/CSV records from CIC-IDS2017 and UNSW-NB15 should be downloaded independently and placed into `data/raw/` for complete model training and benchmarking.
