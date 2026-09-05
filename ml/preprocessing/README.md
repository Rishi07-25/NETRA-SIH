# Data Preprocessing & Windowing

## Overview
This module transforms raw network records into standardized, time-windowed feature representations required by the downstream machine learning models.

## Components
- `clean_data.py`: Handles NaN/infinite values, schema harmonization, and label encoding.
- `feature_engineering.py`: Computes bidirectional flow ratios, packet entropy, and statistical distributions.
- `create_time_windows.py`: Slices continuous network flows into discrete rolling temporal windows ($W$).

## Primary Owner
**Workstream 2: Data & Feature Engineering**
