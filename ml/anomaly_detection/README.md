# Behavioral Anomaly Detection Module

## Overview
This module trains and deploys unsupervised anomaly detection models to detect abnormal traffic deviations from benign baseline behavior.

## Methodology & Zero-Target Contamination Rule
> **CRITICAL SECURITY & METHODOLOGICAL RULE:**
> 
> The Isolation Forest anomaly detector is fitted **strictly on unlabeled feature data $X$**. Target labels $y$ are never passed to the training function. This guarantees that the anomaly detector models purely the geometric structure and dispersion of network feature space without supervisory leakage.
> 
> Labels are referenced exclusively in `ml/evaluation/` for post-training assessment of precision, recall, and false positive rates.

## Components
- `train.py`: Unsupervised `IsolationForest` training pipeline with configurable contamination parameter (default: 0.15).
- `predict.py`: Anomaly scoring interface (`AnomalyDetectorPredictor` and `predict_anomaly`).

## Prediction Output Contract
```json
{
  "model_name": "IsolationForestAnomalyDetector",
  "is_anomalous": true,
  "raw_score": -0.0842,
  "display_score": 0.6038
}
```
*Note: `display_score` is a monotonic sigmoid-transformed representation in $[0, 1]$ designed for UI indicators; it is an operational severity index, NOT a calibrated statistical probability.*

## Primary Owner
**Workstream 1: AI/ML**
