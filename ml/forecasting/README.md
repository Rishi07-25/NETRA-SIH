# Temporal Risk Forecasting

## Overview
The forecasting engine models sequential flow trajectories over rolling time windows to project future attack probabilities and escalation velocity before an attack matures.

## Components
- `train.py`: Training routines for temporal sequence models (LSTMs, GRUs, TCNs, or autoregressive ensemble forecasters).
- `predict.py`: Multi-horizon risk forecasting pipeline ($t + 1\text{m}, t + 5\text{m}, t + 15\text{m}$).

## Primary Owner
**Workstream 1: AI/ML**
