"""Comprehensive evaluation harness across NETRA Threat Engine models.

Evaluates the Logistic Regression baseline, Random Forest primary classifier,
and Isolation Forest anomaly detector. Extracts feature importances and writes
machine-readable result JSONs and confusion matrix CSVs.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.pipeline import Pipeline

from ml.evaluation.metrics import compute_anomaly_metrics, compute_classification_metrics

logger = logging.getLogger(__name__)


def evaluate_models(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    baseline_model: Pipeline,
    primary_model: RandomForestClassifier,
    anomaly_model: IsolationForest,
    feature_names: list[str],
    results_dir: Optional[Union[str, Path]] = "ml/evaluation/results",
    is_synthetic_data: bool = True,
) -> Dict[str, Any]:
    """Run evaluation across all models on held-out test data.

    Parameters
    ----------
    X_test : pd.DataFrame
        Held-out test features.
    y_test : pd.Series
        Ground truth test labels.
    baseline_model : Pipeline
        Trained Logistic Regression pipeline.
    primary_model : RandomForestClassifier
        Trained Random Forest classifier.
    anomaly_model : IsolationForest
        Trained Isolation Forest model.
    feature_names : list[str]
        List of feature column names.
    results_dir : Optional[str or Path]
        Directory to export evaluation metrics and confusion matrices.
    is_synthetic_data : bool
        Flag explicitly documenting synthetic vs benchmark status.

    Returns
    -------
    Dict[str, Any]
        Aggregated evaluation results dictionary.
    """
    logger.info(f"Running model evaluation on {len(X_test)} test samples...")
    classes = sorted(list(y_test.unique()))

    # 1. Evaluate Logistic Regression Baseline
    y_pred_baseline = baseline_model.predict(X_test)
    baseline_metrics = compute_classification_metrics(y_test, y_pred_baseline, labels=classes)

    # 2. Evaluate Primary Random Forest
    y_pred_primary = primary_model.predict(X_test)
    primary_metrics = compute_classification_metrics(y_test, y_pred_primary, labels=classes)

    # 3. Evaluate Isolation Forest (Unsupervised model tested against ground truth)
    raw_anomaly_preds = anomaly_model.predict(X_test)
    is_anomalous_preds = [bool(p == -1) for p in raw_anomaly_preds]
    anomaly_metrics = compute_anomaly_metrics(y_test, is_anomalous_preds)

    # 4. Extract Feature Importances from Random Forest
    importances = primary_model.feature_importances_
    fi_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances,
    }).sort_values(by="importance", ascending=False).reset_index(drop=True)

    # 5. Export Results if directory specified
    if results_dir:
        out_dir = Path(results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meta_header = {
            "dataset_status": (
                "DEMO / SYNTHETIC DATA — NOT REPRESENTATIVE OF REAL-WORLD MODEL PERFORMANCE"
                if is_synthetic_data
                else "BENCHMARK DATASET"
            ),
            "test_samples": len(X_test),
            "num_features": len(feature_names),
            "classes": classes,
        }

        # Classification results JSON
        cls_output = {
            "metadata": meta_header,
            "baseline_logistic_regression": baseline_metrics,
            "primary_random_forest": primary_metrics,
        }
        with open(out_dir / "classification_results.json", "w") as f:
            json.dump(cls_output, f, indent=2)

        # Anomaly results JSON
        anom_output = {
            "metadata": meta_header,
            "isolation_forest": anomaly_metrics,
        }
        with open(out_dir / "anomaly_results.json", "w") as f:
            json.dump(anom_output, f, indent=2)

        # Feature importance CSV
        fi_df.to_csv(out_dir / "feature_importance.csv", index=False)

        # Primary confusion matrix CSV
        cm_df = pd.DataFrame(
            primary_metrics["confusion_matrix"],
            index=[f"true_{c}" for c in classes],
            columns=[f"pred_{c}" for c in classes],
        )
        cm_df.to_csv(out_dir / "confusion_matrix.csv")
        logger.info(f"Evaluation artifacts saved to: {out_dir}")

    return {
        "baseline": baseline_metrics,
        "primary": primary_metrics,
        "anomaly": anomaly_metrics,
        "feature_importance": fi_df,
    }


def run_evaluation():
    """Placeholder interface to satisfy module exports."""
    raise NotImplementedError("Use evaluate_models with trained estimators.")
