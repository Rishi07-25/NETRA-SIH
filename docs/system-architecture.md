# System Architecture

## Modular Components
NETRA is structured as a decoupled, multi-tier distributed architecture designed for low-latency inference and high-throughput network stream processing.

```text
┌─────────────────────────────────────────────────────────────┐
│                      DATA INGESTION                         │
│  - Raw PCAP Files (Offline validation)                      │
│  - Live Packet Sniffer / Synthetic Traffic Injector         │
│  - Flow Extractor (Bidirectional statistical features)      │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   PIPELINE & ML SERVICES                    │
│  - Rolling Window Aggregator (Time-series formulation)      │
│  - Anomaly Detector (Isolation Forest / Autoencoders)       │
│  - Threat Classifier (XGBoost / Random Forest)              │
│  - Risk Forecaster (Temporal Deep Learning / Sequence Model)│
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                     BACKEND SERVICES                        │
│  - FastAPI Application Layer                                │
│  - Threat Scoring & Alert Correlation Engine                │
│  - RESTful APIs / WebSocket Event Stream                    │
│  - Relational Database (PostgreSQL/SQLite)                  │
└──────────────────────────────┬──────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   ANALYST PRESENTATION LAYER                │
│  - React SOC Dashboard                                      │
│  - Threat Score Gauges & Risk Trajectory Timeline           │
│  - Active Attack Table & Event Log Stream                   │
└─────────────────────────────────────────────────────────────┘
```

## Communications & Interfaces
- **Ingestion to ML:** In-memory streaming buffers or parquet/CSV batch pipelines.
- **ML to Backend:** High-performance Python service wrappers invoking serialized models (`.joblib` / `.pt`).
- **Backend to Frontend:** JSON RESTful endpoints and asynchronous WebSocket feeds for sub-second alert updates.
