# Attack Classification Module

## Overview
This module classifies suspicious network behaviors into recognized threat categories (such as `BENIGN`, `Reconnaissance`, `Brute Force`, and `DDoS`).

## Components
- `baseline.py`: Logistic Regression benchmark model with `StandardScaler` and `class_weight='balanced'`. Serves as the conventional baseline required for SIH evaluations.
- `train.py`: Primary Random Forest attack classifier (`n_estimators=200`, `random_state=42`).
- `predict.py`: Standardized prediction interface (`AttackClassifierPredictor` and `predict_attack_class`) validating feature schemas and exposing full class probability distributions.

## Usage
```bash
# Train Logistic Regression Baseline
python -m ml.attack_classification.baseline --features data/features/features_X.csv --labels data/features/labels_y.csv

# Train Primary Random Forest Classifier
python -m ml.attack_classification.train --features data/features/features_X.csv --labels data/features/labels_y.csv
```

## Prediction Interface Output Contract
```json
{
  "model_name": "RandomForestClassifier",
  "predicted_attack": "Reconnaissance",
  "confidence": 0.945,
  "class_probabilities": {
    "BENIGN": 0.025,
    "Brute Force": 0.015,
    "DDoS": 0.015,
    "Reconnaissance": 0.945
  }
}
```

## Primary Owner
**Workstream 1: AI/ML**
