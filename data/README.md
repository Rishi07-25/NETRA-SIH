# Data Directory

## Overview
This directory stores datasets utilized for training, evaluating, and testing NETRA's machine learning pipelines.

## Directory Structure
- `raw/`: Unaltered, original external network datasets (e.g., CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15).
- `processed/`: Cleaned and standardized data with handling for missing values, infinite floats, and deduplication.
- `features/`: Engineered feature representations and time-windowed aggregated matrices ready for model consumption.
- `samples/`: Lightweight sample extracts (<1MB) used for local debugging and CI/CD validation.

## Storage Guidelines
> **IMPORTANT:** Do NOT commit large data files (`.csv`, `.pcap`, `.parquet`) to Git. Large datasets should be stored externally and downloaded via data preparation scripts. Only `.gitkeep` files are tracked in version control.

## Primary Owner
**Workstream 2: Data & Feature Engineering**
