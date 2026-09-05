# Data Schemas & Contracts

## Overview
This package defines the Pydantic data schemas used across the API to guarantee request validation, serialization, and type safety between services.

## Schemas
- `prediction.py`: Request/response schemas for threat classification and confidence outputs.
- `forecast.py`: Schemas for sequential time-window inputs and multi-horizon risk forecasts.
- `network.py`: Schemas for network flow telemetry, throughput statistics, and packet characteristics.

## Primary Owner
**Workstream 4: Backend/API**
