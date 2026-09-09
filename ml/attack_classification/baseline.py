"""Baseline Logistic Regression classifier for NETRA Stage 2.

Serves as the conventional linear benchmark model against which primary
tree/ensemble classifiers are evaluated under identical data splits.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ml.data_validation import load_and_validate_features, split_threat_data

logger = logging.getLogger(__name__)


def build_logistic_regression_pipeline(
    random_state: int = 42,
    max_iter: int = 1000,
    C: float = 1.0,
    class_weight: Optional[str] = "balanced",
) -> Pipeline:
    """Construct scikit-learn Pipeline with StandardScaler and LogisticRegression."""
    return Pipeline([
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(
                random_state=random_state,
                max_iter=max_iter,
                C=C,
                class_weight=class_weight,
                solver="lbfgs",
            ),
        ),
    ])


def train_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = 42,
    max_iter: int = 1000,
    save_path: Optional[Union[str, Path]] = "ml/models/logistic_regression_baseline.joblib",
) -> Tuple[Pipeline, Dict[str, Any]]:
    """Train the Logistic Regression baseline model on training split.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target labels.
    random_state : int
        Deterministic random seed.
    max_iter : int
        Maximum optimization iterations for solver convergence.
    save_path : Optional[str or Path]
        Path to save the serialized model bundle artifact.

    Returns
    -------
    Tuple[Pipeline, Dict[str, Any]]
        Fitted pipeline and model metadata artifact bundle.
    """
    logger.info(f"Training Logistic Regression baseline on {len(X_train)} samples...")
    feature_names = list(X_train.columns)
    classes = sorted(list(y_train.unique()))

    pipeline = build_logistic_regression_pipeline(
        random_state=random_state,
        max_iter=max_iter,
        class_weight="balanced",
    )
    pipeline.fit(X_train, y_train)

    artifact_bundle: Dict[str, Any] = {
        "model_name": "LogisticRegressionBaseline",
        "pipeline": pipeline,
        "feature_names": feature_names,
        "classes": list(pipeline.named_steps["classifier"].classes_),
        "random_state": random_state,
        "scaler_fitted": True,
        "version": "0.2.0",
    }

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact_bundle, out_path)
        logger.info(f"Baseline model bundle saved to: {out_path}")

    return pipeline, artifact_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NETRA Logistic Regression Baseline")
    parser.add_argument("--features", default="data/features/features_X.csv", help="Path to features CSV")
    parser.add_argument("--labels", default="data/features/labels_y.csv", help="Path to labels CSV")
    parser.add_argument("--output", default="ml/models/logistic_regression_baseline.joblib", help="Output model path")
    args = parser.parse_args()

    X, y = load_and_validate_features(args.features, args.labels)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y)
    train_baseline(X_tr, y_tr, save_path=args.output)
