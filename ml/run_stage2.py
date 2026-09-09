"""Stage 2 End-to-End Orchestrator for NETRA AI Threat Engine.

Executes the complete Stage 2 pipeline:
1. Data loading and validation
2. Class distribution inspection
3. Deterministic train/test split
4. Logistic Regression baseline training
5. Random Forest primary classifier training
6. Unsupervised Isolation Forest anomaly training
7. Comprehensive model evaluation and feature importance extraction
8. Model artifact persistence
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from ml.anomaly_detection.train import train_anomaly_detector
from ml.attack_classification.baseline import train_baseline
from ml.attack_classification.train import train_attack_classifier
from ml.data_validation import load_and_validate_features, split_threat_data
from ml.evaluation.evaluate import evaluate_models

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NETRA-ThreatEngine")


def run_stage2_threat_engine(
    features_path: str = "data/features/features_X.csv",
    labels_path: str = "data/features/labels_y.csv",
    models_dir: str = "ml/models",
    results_dir: str = "ml/evaluation/results",
    test_size: float = 0.20,
    random_state: int = 42,
    is_synthetic: bool = True,
):
    """Orchestrate the end-to-end Stage 2 model training and evaluation pipeline."""
    print("=" * 60)
    print("NETRA STAGE 2 — AI THREAT ENGINE")
    print("=" * 60)

    # 1. Load & Validate Features
    logger.info("Loading and validating feature and label datasets...")
    X, y = load_and_validate_features(features_path, labels_path)
    feature_names = list(X.columns)
    total_samples = len(X)
    num_features = len(feature_names)
    unique_classes = sorted(list(y.unique()))
    num_classes = len(unique_classes)

    print(f"\nDataset: {features_path}")
    print(f"Total Samples: {total_samples}")
    print(f"Features ({num_features}): {feature_names}")
    print(f"Classes ({num_classes}): {unique_classes}")

    if is_synthetic:
        print("\n" + "!" * 60)
        print("DEMO / SYNTHETIC DATA — NOT REPRESENTATIVE OF REAL-WORLD MODEL PERFORMANCE")
        print("!" * 60)

    # 2. Inspect Class Distribution
    print("\n--- CLASS DISTRIBUTION ---")
    counts = y.value_counts()
    for cls_name, count in counts.items():
        pct = (count / total_samples) * 100
        print(f"  {cls_name:<20}: {count:>4} samples ({pct:>5.1f}%)")

    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = split_threat_data(
        X, y, test_size=test_size, random_state=random_state
    )
    print(f"\nTrain Split: {len(X_train)} samples | Test Split: {len(X_test)} samples")

    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)

    # 4. Train Baseline (Logistic Regression)
    print("\n------------------------------------------------------------")
    print("TRAINING BASELINE — LOGISTIC REGRESSION")
    print("------------------------------------------------------------")
    baseline_file = models_path / "logistic_regression_baseline.joblib"
    baseline_model, _ = train_baseline(
        X_train, y_train, random_state=random_state, save_path=baseline_file
    )

    # 5. Train Primary Classifier (Random Forest)
    print("\n------------------------------------------------------------")
    print("TRAINING PRIMARY CLASSIFIER — RANDOM FOREST")
    print("------------------------------------------------------------")
    rf_file = models_path / "random_forest_classifier.joblib"
    rf_model, _ = train_attack_classifier(
        X_train, y_train, n_estimators=200, random_state=random_state, save_path=rf_file
    )

    # 6. Train Anomaly Detector (Isolation Forest - Unsupervised, X_train only)
    print("\n------------------------------------------------------------")
    print("TRAINING ANOMALY DETECTOR — ISOLATION FOREST")
    print("------------------------------------------------------------")
    anom_file = models_path / "isolation_forest.joblib"
    anom_model, _ = train_anomaly_detector(
        X_train, contamination=0.15, n_estimators=150, random_state=random_state, save_path=anom_file
    )

    # 7. Comprehensive Model Evaluation
    print("\n------------------------------------------------------------")
    print("EVALUATING MODELS ON TEST SPLIT")
    print("------------------------------------------------------------")
    results = evaluate_models(
        X_test=X_test,
        y_test=y_test,
        baseline_model=baseline_model,
        primary_model=rf_model,
        anomaly_model=anom_model,
        feature_names=feature_names,
        results_dir=results_dir,
        is_synthetic_data=is_synthetic,
    )

    b_metrics = results["baseline"]
    p_metrics = results["primary"]
    a_metrics = results["anomaly"]
    fi_df = results["feature_importance"]

    print("\n--- BASELINE (LOGISTIC REGRESSION) METRICS ---")
    print(f"Accuracy     : {b_metrics['accuracy']:.4f}")
    print(f"Macro F1     : {b_metrics['f1_macro']:.4f}")
    print(f"Weighted F1  : {b_metrics['f1_weighted']:.4f}")

    print("\n--- PRIMARY (RANDOM FOREST) METRICS ---")
    print(f"Accuracy     : {p_metrics['accuracy']:.4f}")
    print(f"Macro F1     : {p_metrics['f1_macro']:.4f}")
    print(f"Weighted F1  : {p_metrics['f1_weighted']:.4f}")

    print("\n--- ANOMALY DETECTOR (ISOLATION FOREST) METRICS ---")
    print(f"Precision    : {a_metrics['anomaly_precision']:.4f}")
    print(f"Recall       : {a_metrics['anomaly_recall']:.4f}")
    print(f"F1-Score     : {a_metrics['anomaly_f1']:.4f}")
    print(f"False Pos Rate: {a_metrics['false_positive_rate']:.4f}")

    print("\n--- TOP 10 FEATURE IMPORTANCE (RANDOM FOREST) ---")
    for idx, row in fi_df.head(10).iterrows():
        print(f"  {idx + 1:>2}. {row['feature']:<25}: {row['importance']:.4f}")

    print("\n------------------------------------------------------------")
    print("ARTIFACTS SUMMARY")
    print("------------------------------------------------------------")
    print(f"Models directory    : {models_path}")
    print(f"Results directory   : {results_dir}")
    print("=" * 60)
    print("NETRA STAGE 2 THREAT ENGINE EXECUTION COMPLETE")
    print("=" * 60)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NETRA Stage 2 AI Threat Engine")
    parser.add_argument("--features", default="data/features/features_X.csv", help="Path to features CSV")
    parser.add_argument("--labels", default="data/features/labels_y.csv", help="Path to labels CSV")
    parser.add_argument("--models-dir", default="ml/models", help="Directory to save model artifacts")
    parser.add_argument("--results-dir", default="ml/evaluation/results", help="Directory to save results")
    parser.add_argument("--test-size", type=float, default=0.20, help="Test split proportion")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed")
    parser.add_argument("--benchmark", action="store_true", help="Flag if running real benchmark dataset")
    args = parser.parse_args()

    run_stage2_threat_engine(
        features_path=args.features,
        labels_path=args.labels,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        is_synthetic=not args.benchmark,
    )
