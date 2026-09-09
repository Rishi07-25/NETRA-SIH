"""Primary Supervised Attack Classifier for NETRA Stage 2 Threat Engine.

Trains a robust Random Forest classifier, extracts feature importance,
supports multiclass classification with probability outputs, and serializes
complete model artifact bundles preserving feature schema and label mappings.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ml.data_validation import load_and_validate_features, split_threat_data

logger = logging.getLogger(__name__)


def train_attack_classifier(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 200,
    random_state: int = 42,
    max_depth: Optional[int] = None,
    class_weight: Optional[str] = "balanced",
    save_path: Optional[Union[str, Path]] = "ml/models/random_forest_classifier.joblib",
) -> Tuple[RandomForestClassifier, Dict[str, Any]]:
    """Train the primary Random Forest attack classification model.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training feature matrix.
    y_train : pd.Series
        Training target attack labels.
    n_estimators : int
        Number of decision trees in the forest.
    random_state : int
        Deterministic random seed.
    max_depth : Optional[int]
        Maximum tree depth.
    class_weight : Optional[str]
        Strategy to address class imbalance.
    save_path : Optional[str or Path]
        Path to serialize the trained model bundle.

    Returns
    -------
    Tuple[RandomForestClassifier, Dict[str, Any]]
        Fitted model and artifact bundle containing model and schema metadata.
    """
    logger.info(f"Training Random Forest classifier on {len(X_train)} samples with {X_train.shape[1]} features...")
    feature_names = list(X_train.columns)

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        class_weight=class_weight,
        n_jobs=-1,
    )
    clf.fit(X_train, y_train)

    artifact_bundle: Dict[str, Any] = {
        "model_name": "RandomForestClassifier",
        "model": clf,
        "feature_names": feature_names,
        "classes": list(clf.classes_),
        "random_state": random_state,
        "n_estimators": n_estimators,
        "class_weight": class_weight,
        "version": "0.2.0",
    }

    if save_path:
        out_path = Path(save_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(artifact_bundle, out_path)
        logger.info(f"Random Forest model bundle saved to: {out_path}")

    return clf, artifact_bundle


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NETRA Random Forest Classifier")
    parser.add_argument("--features", default="data/features/features_X.csv", help="Path to features CSV")
    parser.add_argument("--labels", default="data/features/labels_y.csv", help="Path to labels CSV")
    parser.add_argument("--output", default="ml/models/random_forest_classifier.joblib", help="Output model path")
    args = parser.parse_args()

    X, y = load_and_validate_features(args.features, args.labels)
    X_tr, X_te, y_tr, y_te = split_threat_data(X, y)
    train_attack_classifier(X_tr, y_tr, save_path=args.output)
