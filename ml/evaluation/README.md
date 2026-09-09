# Model Evaluation & Metrics

## Overview
This directory contains standardized evaluation harnesses, metric computations, and benchmark results across NETRA's threat detection models.

## Metrics Computed
- **Multiclass Attack Classification:**
  - Accuracy
  - Macro F1-score & Weighted F1-score (critical for imbalanced cybersecurity distributions)
  - Precision & Recall per attack category
  - Confusion Matrix (`confusion_matrix.csv`)
- **Unsupervised Anomaly Detection:**
  - Anomaly Precision & Recall (assessed post-training against non-benign ground-truth labels)
  - False Positive Rate (FPR) on benign flows
- **Feature Importance:**
  - Gini importance rankings from Random Forest saved to `feature_importance.csv`.

## Artifact Storage
Evaluation outputs are written to `ml/evaluation/results/`:
- `classification_results.json`: Baseline vs. Primary performance comparisons.
- `anomaly_results.json`: Isolation Forest benchmark metrics.
- `feature_importance.csv`: Descending ranking of feature predictive power.
- `confusion_matrix.csv`: Full class attribution matrix.

## Primary Owner
**Workstream 1: AI/ML**
