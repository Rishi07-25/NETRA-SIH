# Data Directory

## Overview
This directory stores datasets utilized for training, evaluating, and testing NETRA's machine learning pipelines.

## Directory Structure
- `raw/`: Unaltered, original external network datasets (e.g., CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15). Large files in this directory are git-ignored.
- `processed/`: Cleaned and standardized datasets produced by `ml/preprocessing/clean_data.py`. Infinite values are imputed, duplicates removed, and labels canonicalized.
- `features/`: Engineered tabular feature representations (`features_X.csv`, `labels_y.csv`) and time-windowed aggregated matrices (`time_windows.csv`) ready for model consumption.
- `samples/`: Lightweight sample extracts (<1MB) used for local debugging and continuous testing. Contains `synthetic_network_flows.csv`.

## Storage & Version Control Guidelines
> **CRITICAL RULE:** Do NOT commit large data files (`.csv`, `.pcap`, `.pcapng`, `.parquet`) to Git. 
> 
> Real-world benchmark datasets (such as CIC-IDS2017 and CSE-CIC-IDS2018) span tens of gigabytes. Committing large binaries exhausts GitHub repository quotas, slows down developer clones, and causes merge overhead. Only `.gitkeep` placeholders and small development samples (`data/samples/synthetic_network_flows.csv`) are tracked. Real raw datasets should be downloaded locally using data staging scripts.

## Primary Owner
**Workstream 2: Data & Feature Engineering**
