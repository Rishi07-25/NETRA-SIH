"""Inference interface for NETRA behavioral anomaly detection.

Outputs raw decision scores and normalized display scores without confusing
heuristic normalization with calibrated probability distributions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import joblib
import numpy as np
import pandas as pd

from ml.data_validation import validate_feature_data

logger = logging.getLogger(__name__)


class AnomalyDetectorPredictor:
    """Predictor wrapper for NETRA Isolation Forest anomaly detection models."""

    def __init__(self, model_bundle_or_path: Union[str, Path, Dict[str, Any]]):
        if isinstance(model_bundle_or_path, (str, Path)):
            bundle_path = Path(model_bundle_or_path)
            if not bundle_path.exists():
                raise FileNotFoundError(f"Anomaly model artifact not found at {bundle_path}")
            self.bundle: Dict[str, Any] = joblib.load(bundle_path)
        else:
            self.bundle = model_bundle_or_path

        self.model = self.bundle.get("model")
        if self.model is None:
            raise ValueError("Bundle does not contain a valid 'model' estimator.")

        self.feature_names: List[str] = self.bundle["feature_names"]
        self.contamination: float = self.bundle.get("contamination", 0.15)
        self.model_name: str = self.bundle.get("model_name", "IsolationForestAnomalyDetector")

    def predict_single(self, feature_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Score a single network flow sample for anomalous behavioral deviation.

        Parameters
        ----------
        feature_input : Dict, pd.Series, or 1-row pd.DataFrame
            Feature vector attributes.

        Returns
        -------
        Dict[str, Any]
            Contains:
            - is_anomalous: Boolean flag (-1 -> True, 1 -> False from IsolationForest)
            - raw_score: Raw decision_function output (negative indicates outlier)
            - display_score: Monotonic normalized score in [0, 1] for UI visualization
        """
        if isinstance(feature_input, dict):
            df_in = pd.DataFrame([feature_input])
        elif isinstance(feature_input, pd.Series):
            df_in = pd.DataFrame([feature_input.to_dict()])
        elif isinstance(feature_input, pd.DataFrame):
            if len(feature_input) != 1:
                raise ValueError(f"predict_single expects exactly 1 row, got {len(feature_input)}")
            df_in = feature_input.copy()
        else:
            raise TypeError(f"Unsupported feature_input type: {type(feature_input)}")

        df_val, _ = validate_feature_data(df_in, expected_features=self.feature_names)

        # Scikit-learn IsolationForest: 1 = inlier, -1 = outlier
        raw_pred = int(self.model.predict(df_val)[0])
        is_anomalous = bool(raw_pred == -1)

        # Raw decision score: higher = more normal, lower/negative = more anomalous
        raw_score = float(self.model.decision_function(df_val)[0])

        # Display score transformation: sigmoid-style monotonic mapping to [0, 1]
        # Higher display score = higher anomaly severity
        display_score = float(1.0 / (1.0 + np.exp(raw_score * 5.0)))

        return {
            "model_name": self.model_name,
            "is_anomalous": is_anomalous,
            "raw_score": raw_score,
            "display_score": round(display_score, 4),
        }

    def predict_batch(self, df_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """Score a batch of network flow vectors."""
        df_val, _ = validate_feature_data(df_features, expected_features=self.feature_names)
        raw_preds = self.model.predict(df_val)
        raw_scores = self.model.decision_function(df_val)

        results = []
        for pred, score in zip(raw_preds, raw_scores):
            is_anom = bool(pred == -1)
            display_score = float(1.0 / (1.0 + np.exp(score * 5.0)))
            results.append({
                "is_anomalous": is_anom,
                "raw_score": float(score),
                "display_score": round(display_score, 4),
            })
        return results


def predict_anomaly(
    feature_vector: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model_path: Union[str, Path] = "ml/models/isolation_forest.joblib",
) -> Dict[str, Any]:
    """Convenience functional interface for anomaly prediction."""
    predictor = AnomalyDetectorPredictor(model_path)
    return predictor.predict_single(feature_vector)
