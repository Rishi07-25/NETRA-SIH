# Backend API Services

## Overview
The `backend/` directory houses the FastAPI application that serves real-time inference, risk forecasts, behavioral anomaly detection, network telemetry, and security events to the SOC frontend dashboard.

## Capabilities & Endpoints
The backend exposes modular REST endpoints supporting both direct path access and `/api/v1` namespace:
- `GET /`: Health check and system operational capabilities report.
- `POST /predict/` (or `/api/v1/predict/`): Threat classification powered by Random Forest and feature attribution.
- `POST /anomaly/` (or `/api/v1/anomaly/`): Behavioral anomaly scoring via unsupervised Isolation Forest.
- `POST /forecast/` (or `/api/v1/forecast/`): Multi-horizon attack risk and trajectory forecasting powered by the Temporal World Model.
- `GET /network/stats` (or `/api/v1/network/stats`): Real-time network throughput, active flows, and anomaly rates.
- `GET /events/` (or `/api/v1/events/`): Security event feed and early-warning alerts.

## Directory Structure
- `main.py`: Application entry point, ASGI configuration, CORS setup, and router registration.
- `api/`: REST API endpoint handlers (`predict.py`, `forecast.py`, `anomaly.py`, `network.py`, `events.py`).
- `services/`: Business logic and ML model wrappers (`prediction_service.py`, `forecasting_service.py`, `anomaly_service.py`, `traffic_service.py`).
- `schemas/`: Pydantic data models for request payload validation and response contracts (`prediction.py`, `forecast.py`, `network.py`, `anomaly.py`, `events.py`).
- `tests/`: Automated unit and API integration tests (`test_api.py`).

## Running Locally
```bash
# Start backend server with uvicorn
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```
