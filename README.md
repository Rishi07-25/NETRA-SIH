# NETRA: Network Threat Early-warning & Risk Analytics

[![SIH Problem Statement](https://img.shields.io/badge/SIH26153-Network%20Attack%20Forecasting-blue.svg)](https://www.sih.gov.in/)
[![Organization](https://img.shields.io/badge/Organization-NTRO-red.svg)](https://ntro.gov.in/)
[![Status](https://img.shields.io/badge/Status-🚧%20Initial%20Repository%20Setup-yellow.svg)](#development-status)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> **SIH26153 — AI based Network Attack Forecasting from Network Traffic Data**  
> **Organization:** National Technical Research Organisation (NTRO)  
> **Theme:** Blockchain & Cybersecurity  
> **Category:** Software  

---

## 🎯 Vision

> *"Don't just detect an attack after it happens — forecast the likelihood of an attack before it happens."*

Conventional network security defenses (IDS, IPS, SIEM) are inherently **reactive**: they identify anomalies or trigger signatures only after an intrusion has commenced or succeeded. 

**NETRA (Network Threat Early-warning & Risk Analytics)** shifts network defense from post-incident response to proactive, temporal threat forecasting. By continuously evaluating sequential telemetry, traffic flow kinetics, and preparatory patterns (such as reconnaissance, horizontal probes, and credential testing), NETRA forecasts increasing attack risk vectors before catastrophic compromise occurs.

---

## 🔬 Problem Statement & Context

Modern cyber adversaries do not execute intrusions instantaneously. Most major breaches follow structured cyber kill-chains—spanning stealth reconnaissance, resource staging, lateral reconnaissance, and privilege escalation prior to exfiltration or disruption.

Traditional monitoring systems either miss these sub-threshold indicators or flood Security Operations Centers (SOC) with unranked, context-free alerts. NETRA analyzes continuous network traffic flow behaviors over dynamic time windows to forecast attack probability, compute holistic threat scores, and deliver actionable early warnings to analysts.

---

## 🔄 Proposed Pipeline

```text
Network Traffic (PCAP / Live Flow Telemetry)
       │
       ▼
Traffic / Flow Processing & Ingestion
       │
       ▼
Feature Engineering & Window Aggregation
       │
       ▼
Behavioral Anomaly Detection (Unsupervised Baseline)
       │
       ▼
Attack Classification (Multi-class Threat Attribution)
       │
       ▼
Temporal Risk Forecasting (Time-series / Sequence Modeling)
       │
       ▼
Dynamic Threat Score Formulation
       │
       ▼
SOC Dashboard & Early-Warning Dispatch
```

---

## ⚡ Core Capabilities

- **Network Traffic Analysis:** Ingest and parse flow-level metrics, connection patterns, and statistical distributions.
- **Behavioral Anomaly Detection:** Identify baseline deviations and suspicious traffic anomalies without relying solely on static signatures.
- **Attack Classification:** Granular categorization of attack families (e.g., Reconnaissance, Brute Force, DoS/DDoS).
- **Temporal Attack-Risk Forecasting:** Predict upcoming risk surges and escalation trajectories across rolling future time windows.
- **Threat Scoring:** Unified scoring engine summarizing network posture, asset criticality, and imminent exposure.
- **Early Warning System:** Prioritized alerts and risk flags before high-impact attack execution occurs.
- **SOC-Style Visualization:** Intuitive, mission-critical operational dashboard for security analysts and incident responders.
- **Explainable Security Insights (Future Scope):** Flow-level attribution and feature interpretability to explain forecasting decisions.

---

## 🛠 Technology Stack

The planned technology stack across project stages includes:

- **Data Processing & Analytics:** Python, Pandas, NumPy
- **Machine Learning & Modeling:** Scikit-learn, XGBoost, Time-series sequence forecasting architectures
- **Backend Services & API:** FastAPI, Uvicorn, Pydantic
- **Frontend / SOC Interface:** React, Tailwind CSS
- **Data Persistence:** PostgreSQL / SQLite
- **Environment & Deployment:** Docker, Docker Compose

*(Note: The stack above reflects initial planning; individual modules are added incrementally per the roadmap).*

---

## 📂 Repository Structure

The NETRA repository is organized cleanly by domain responsibilities:

```text
NETRA/
├── docs/             # Technical specifications, architecture designs, research & guides
├── data/             # Raw, processed, feature sets, and sample data (git-ignored artifacts)
├── ml/               # Machine learning pipelines: preprocessing, anomaly, classification, forecasting
├── network/          # Traffic generators, packet capture modules, attack scenario configs
├── backend/          # FastAPI application, route handlers, core services, schemas, tests
├── frontend/         # React SOC dashboard, visualization components, API client
├── demo/             # Scripted scenarios, sample artifacts, and demonstration assets
├── presentation/     # Presentation decks, architecture diagrams, and speaker notes
├── deployment/       # Dockerfiles, Nginx reverse proxy configs, and deployment scripts
└── tests/            # End-to-end and integration test suites
```

Detailed directory specifications:
- [`docs/`](docs/README.md): Architecture designs, problem statement breakdown, AI methodology, and judge QA.
- [`data/`](data/README.md): Directory guidelines and storage practices for network datasets.
- [`ml/`](ml/README.md): End-to-end ML codebase across anomaly detection, classification, and forecasting.
- [`network/`](network/README.md): Synthetic traffic generators, PCAP parsers, and scenario profiles.
- [`backend/`](backend/README.md): RESTful backend delivering inference endpoints and system telemetry.
- [`frontend/`](frontend/README.md): Interactive SOC analyst web application.
- [`demo/`](demo/README.md): Demonstration plans and evaluation playbooks.
- [`presentation/`](presentation/README.md): SIH presentation slides and visual assets.
- [`deployment/`](deployment/README.md): Containerization and production orchestration configurations.
- [`tests/`](tests/README.md): Integration and end-to-end system testing harness.

---

## 👥 Team Structure & Workstreams

To ensure collaborative and concurrent development across the 6-member team, work is divided into six specialized workstreams:

| Workstream | Focus Area | Primary Directories |
|---|---|---|
| **1. AI/ML** | Anomaly detection, classification models, temporal forecasting | `ml/anomaly_detection/`, `ml/attack_classification/`, `ml/forecasting/` |
| **2. Data & Feature Engineering** | Data wrangling, windowing, flow normalization, feature selection | `data/`, `ml/preprocessing/`, `ml/notebooks/` |
| **3. Network Security** | Traffic simulation, PCAP generation, threat scenario modeling | `network/traffic_generator/`, `network/packet_capture/`, `network/scenarios/` |
| **4. Backend / API** | Microservices, inference integration, data schemas, API routes | `backend/api/`, `backend/services/`, `backend/schemas/` |
| **5. Frontend / SOC Dashboard** | Real-time dashboard, risk gauges, timeline, alerts UI | `frontend/src/` |
| **6. Integration, Research & Presentation** | End-to-end integration, evaluation benchmarks, SIH deliverables | `docs/`, `demo/`, `presentation/`, `tests/` |

---

## 🚧 Development Status & Roadmap

Current Milestone: **🚧 Initial Repository Setup**

- [ ] Dataset selection & validation (e.g., CIC-IDS2017, CSE-CIC-IDS2018, UNSW-NB15)
- [ ] Data preprocessing & cleaning pipelines
- [ ] Flow-based feature engineering & rolling time-window generation
- [ ] Baseline anomaly detection models
- [ ] Multi-class attack classification models
- [ ] Temporal risk forecasting engine
- [ ] FastAPI backend services & endpoint development
- [ ] SOC dashboard UI & visualization components
- [ ] End-to-end pipeline integration
- [ ] System evaluation & benchmark verification
- [ ] Comprehensive demo rehearsal & scenario packaging
- [ ] SIH final presentation & documentation delivery

---

## 🚀 Getting Started

### Prerequisites

- Python 3.10+
- Git
- Node.js 18+ (for frontend development)
- Docker & Docker Compose (optional for containerized runs)

### Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-org>/NETRA.git
   cd NETRA
   ```

2. **Configure environment:**
   ```bash
   cp .env.example .env
   ```

3. **Install Python dependencies:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
