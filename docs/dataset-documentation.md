# Dataset Documentation

## Benchmark Datasets Under Evaluation
For training, validation, and benchmarking of NETRA's machine learning models, several standard cybersecurity datasets are being analyzed:

1. **CIC-IDS2017 (Canadian Institute for Cybersecurity)**
   - *Description:* Realistic background network traffic intertwined with multiple attack profiles (DDoS, Brute Force, Web Attacks, Infiltration).
   - *Format:* PCAP captures and pre-extracted bidirectional flow CSVs (80+ statistical features).
   
2. **CSE-CIC-IDS2018**
   - *Description:* Large-scale enterprise network simulation featuring modern attack tactics executed on AWS infrastructure.
   - *Format:* Raw PCAP and CICFlowMeter feature extractions.

3. **UNSW-NB15**
   - *Description:* Hybrid collection of real modern normal activities and synthetic contemporary attack behaviors.
   - *Format:* Comprehensive flow records with 49 features.

---

## Data Ingestion & Storage Policy
- **Storage Rules:** Large `.pcap`, `.pcapng`, and raw `.csv` files exceeding 50MB MUST NOT be checked into version control.
- **Directory Layout:**
  - `data/raw/`: Original unaltered datasets (download scripts provided).
  - `data/processed/`: Cleaned, normalized datasets with resolved missing values and scaled attributes.
  - `data/features/`: Window-aggregated feature vectors ready for model consumption.
  - `data/samples/`: Small, synthetic, or truncated samples (<1MB) for testing pipelines.
