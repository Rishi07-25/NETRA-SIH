# Backend Services Layer

## Overview
The `backend/services/` package contains the business logic, orchestration pipelines, and connectors interfacing with ML model artifacts and network stream processors.

## Services
- `prediction_service.py`: Loads serialized classifiers and manages attack category inference.
- `forecasting_service.py`: Orchestrates multi-step temporal risk extrapolation and trend calculations.
- `anomaly_service.py`: Manages baseline comparison and anomaly score normalization.
- `traffic_service.py`: Handles network stream ingestion, flow windowing buffers, and metrics aggregation.

## Primary Owner
**Workstream 4: Backend/API**
