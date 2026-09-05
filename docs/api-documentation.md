# API Documentation Specification

## Overview
NETRA's backend is powered by FastAPI, exposing modular endpoints designed for SOC dashboards and SIEM integrations.

---

## Planned Endpoints

### 1. Health & Root
- **`GET /`**
  - Response: System health, operational state, API version.

### 2. Predictions & Classification
- **`POST /api/v1/predict`**
  - **Input:** Flow feature vector (JSON).
  - **Output:** Predicted attack classification, confidence score, and contributing feature weights.

### 3. Attack Risk Forecasting
- **`POST /api/v1/forecast`**
  - **Input:** Sequence of time-windowed traffic features across horizons $t_0 \dots t_k$.
  - **Output:** Forecasted risk profile across future horizons ($t + 1\text{m}, t + 5\text{m}, t + 15\text{m}$) and escalation probability.

### 4. Anomaly Detection
- **`POST /api/v1/anomaly`**
  - **Input:** Current network flow slice.
  - **Output:** Anomaly flag (`true`/`false`), reconstruction error or outlier metric.

### 5. Network Metrics & Events
- **`GET /api/v1/network/stats`**
  - **Output:** Live packet throughput, active flow count, bandwidth utilization.
- **`GET /api/v1/events`**
  - **Output:** Historical stream of high-severity alerts and forecasted threat surges.
