# Machine Learning (ML) Pipelines & AI Threat Engine

## Overview
The `ml/` directory encapsulates NETRA's multi-stage machine learning architecture. 

### Critical Paradigm Distinction: Stage 2 vs. Stage 3
- **STAGE 2 (AI THREAT ENGINE — CURRENT):**
  - Evaluates **current** flow behavior ($X_t \rightarrow \hat{y}_t$).
  - Consumes flow-level tabular features (`features_X.csv` and `labels_y.csv`).
  - Performs multi-class attack classification and unsupervised behavioral anomaly detection.
  - Combines classification confidence and anomaly telemetry into unified threat level heuristics.
- **STAGE 3 (TEMPORAL ATTACK FORECASTING — FUTURE):**
  - Evaluates **temporal sequences of historical traffic windows** ($[W_{t-k} \dots W_t] \rightarrow \text{Risk}_{t+\Delta t}$).
  - Consumes sequential temporal windows (`time_windows.csv`).
  - Forecasts impending attack risk before full-scale volumetric or intrusion escalation occurs.

---

## Directory Structure
- `data_validation.py`: Reusable validation layer checking schema conformance, numeric validity, and preventing target/identifier leakage.
- `threat_engine.py`: Unified Threat Engine combining supervised classification and unsupervised anomaly detection with transparent heuristic threat levels.
- `run_stage2.py`: End-to-end command-line orchestrator executing validation, splitting, baseline training, primary classification, anomaly detection, and evaluation.
- `attack_classification/`:
  - `baseline.py`: Conventional Logistic Regression benchmark pipeline with `ColumnTransformer` (`OneHotEncoder` for protocol, `StandardScaler` for continuous features).
  - `train.py`: Primary Random Forest attack classifier.
  - `predict.py`: Standardized prediction interface exposing class probabilities, confidence, and schema validation.
- `anomaly_detection/`:
  - `train.py`: Unsupervised Isolation Forest model fitted **strictly on features $X$ without labels**.
  - `predict.py`: Anomaly prediction interface exposing raw decision scores and normalized display scores.
- `forecasting/` (STAGE 3A, 3B.1 & 3B.2):
  - `schema.py`: Canonical schema definitions, column aliases (CIC-IDS2017 & UNSW-NB15), label normalization, and operational attack-stage taxonomy.
  - `dataset_loader.py`: Real-world dataset loader supporting single/multi-CSV/Parquet ingestion, flexible timestamp normalization, and data health auditing.
  - `temporal_state.py`: Formulates aggregate behavioral state $S_t$ over half-open windows $[t, t + \Delta)$ with 19 predictive features and quarantined target metadata.
  - `sequence_builder.py`: Constructs sliding sequence tensors $X_t \in \mathbb{R}^{N \times L \times D}$ paired with temporal window observation metadata.
  - `targets.py`: Extracts future targets $y_{t+H}$ (binary, canonical category, operational attack stage, attack risk score $\in [0, 1]$).
  - `temporal_split.py`: Partition-first chronological and day-based splitting with preprocessing isolation (scaler fitted strictly on training data).
  - `validate_dataset.py`: Comprehensive temporal dataset quality auditor and artifact exporter.
  - `world_model.py` (Stage 3B.1 & 3B.2): PyTorch LSTM temporal world model with 5 heads (binary, attack type, operational stage, risk score, and standardized next state).
  - `train_world_model.py` (Stage 3B.1 & 3B.2): Multi-task training pipeline with state loss, configurable loss weights, benign attribution masking, and checkpoint v2.
  - `predict_world_model.py` (Stage 3B.1 & 3B.2): Disentangled probability, risk score, and state inference interface.
  - `rollout.py` (Stage 3B.2): Autoregressive multi-step state rollout engine, naive persistence baseline, and horizon degradation evaluator.
- `evaluation/`:
  - `metrics.py`: Calculation of multiclass macro/weighted F1, recall, precision, and anomaly FPR.
  - `evaluate.py`: Multi-model evaluation harness exporting JSON metrics, confusion matrices, and feature importance.
  - `results/`: Directory storing generated evaluation artifacts (git-ignored).
- `models/`: Directory storing serialized model bundles (`.joblib`) (git-ignored).

---

## Model Artifact Bundle Format
Every saved model bundle preserves metadata required for schema verification and deterministic inference:
```python
{
    "model_name": "RandomForestClassifier",
    "model": <fitted_estimator>,
    "feature_names": ["dst_port", "protocol", "flow_duration", ...],
    "classes": ["BENIGN", "Brute Force", "DDoS", "Reconnaissance"],
    "random_state": 42,
    "version": "0.2.0"
}
```

---

## Execution Commands

### Run Complete Stage 2 Pipeline
```bash
python -m ml.run_stage2 --features data/features/features_X.csv --labels data/features/labels_y.csv
```

### Run Unit and Integration Tests
```bash
PYTHONPATH=. pytest -v tests/
```

---

## Synthetic Data Warning
> **IMPORTANT:** Evaluation metrics generated on `data/samples/synthetic_network_flows.csv` are for **demonstration and pipeline smoke testing only**. High or perfect classification scores on 50 synthetic rows do NOT represent real-world benchmark performance. Real evaluations will be conducted when full benchmark datasets (CIC-IDS2017, UNSW-NB15) are loaded into `data/raw/`.
