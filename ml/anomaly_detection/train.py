"""Unsupervised Behavioral Anomaly Detection for NETRA Stage 2 Threat Engine.

Trains an Isolation Forest estimator exclusively on unlabeled flow features X.
Labels y are STRICTLY PROHIBITED during fitting to prevent target contamination.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

from ml.data_validation import load_and_validate_features, validate_feature_data

logger = logging.getLogger(__name__)


def train_anomaly_detector(
    X_train: pd.DataFrame,
    contamination: float = 0.15,
    n_estimators: int = 150,
    random_state: int = 42,
    save_path: Optional[Union[str, Path]] = "ml/models/isolation_forest.joblib",
) -> Tuple[IsolationForest, Dict[str, Any]]:
    """Train unsupervised Isolation Forest on network feature matrix.

    CRITICAL SECURITY & METHODOLOGICAL RULE:
    This function accepts ONLY feature matrix X_train.
    Target labels y are NOT accepted to mathematically guarantee zero target leakage.

    Parameters
    ----------
    X_train : pd.DataFrame
        Unlabeled network flow feature matrix.
    contamination : float
        Expected proportion of anomalies in the dataset.
    n_estimators : int
        Number of isolation trees.
    random_state : int
        Deterministic random seed.
    save_path : Optional[str or Path]
        Destination to save model artifact bundle.

    Returns
    -------
    Tuple[IsolationForest, Dict[str, Any]]
        Fitted model and artifact bundle.
    """
    # Validate feature data without any labels
    X_val, _ = validate_feature_data(X_train)
    feature_names = list(X_val.columns)

    logger.info(
        f"Fitting unsupervised Isolation Forest on {len(X_val)} samples (contamination={contamination})..."
    )
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_val)

    artifact_bundle: Dict[str, Any] = {
        "model_name": "IsolationForestAnomalyDetector",
        "model": model,
        "feature_names": feature_names,
        "contamination": contamination,
        "random_state": random_state,
        "offset_": float(model.offset_),
        "version": "0.2.0",
    }

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact_bundle, out_path)
        logger.info(f"Anomaly model bundle saved to: {out_path}")

    return model, artifact_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NETRA Isolation Forest Anomaly Detector")
    parser.add_argument("--features", default="data/features/features_X.csv", help="Path to features CSV")
    parser.add_argument("--contamination", type=float, default=0.15, help="Anomaly contamination fraction")
    parser.add_argument("--output", default="ml/models/isolation_forest.joblib", help="Output model path")
    args = parser.parse_args()

    # Load features only; labels ignored during fitting
    X, _ = load_and_validate_features(args.features, "data/features/labels_y.csv")
    train_anomaly_detector(X, contamination=args.contamination, save_path=args.output)
