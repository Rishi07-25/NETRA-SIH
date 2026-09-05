# Backend API Services

## Overview
The `backend/` directory houses the FastAPI application that serves real-time inference, risk forecasts, and network telemetry to the SOC frontend dashboard.

## Directory Structure
- `main.py`: Application entry point, ASGI configuration, CORS setup, and router registration.
- `api/`: REST API endpoint handlers categorized by capability (`predict`, `forecast`, `anomaly`, `network`, `events`).
- `services/`: Business logic, ML model serialization bridges, and data streaming processors.
- `schemas/`: Pydantic data models for request payload validation and response contracts.
- `tests/`: Automated unit and API integration tests.

## Primary Owner
**Workstream 4: Backend/API**
