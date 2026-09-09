"""Inference interface for NETRA attack classification models.

Loads serialized model bundles, validates input schemas, preserves feature ordering,
and outputs standardized predictions with class probabilities and confidence scores.
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


class AttackClassifierPredictor:
    """Predictor wrapper for NETRA attack classification models."""

    def __init__(self, model_bundle_or_path: Union[str, Path, Dict[str, Any]]):
        if isinstance(model_bundle_or_path, (str, Path)):
            bundle_path = Path(model_bundle_or_path)
            if not bundle_path.exists():
                raise FileNotFoundError(f"Model artifact not found at {bundle_path}")
            self.bundle: Dict[str, Any] = joblib.load(bundle_path)
        else:
            self.bundle = model_bundle_or_path

        # Handle both pipeline (Logistic Regression) and direct model (Random Forest)
        self.model = self.bundle.get("pipeline") or self.bundle.get("model")
        if self.model is None:
            raise ValueError("Model bundle does not contain a valid 'pipeline' or 'model' estimator.")

        self.feature_names: List[str] = self.bundle["feature_names"]
        self.classes: List[str] = list(self.bundle["classes"])
        self.model_name: str = self.bundle.get("model_name", "AttackClassifier")

    def predict_single(self, feature_input: Union[Dict[str, Any], pd.Series, pd.DataFrame]) -> Dict[str, Any]:
        """Generate prediction for a single flow feature vector.

        Parameters
        ----------
        feature_input : Dict, pd.Series, or 1-row pd.DataFrame
            Incoming network flow feature attributes.

        Returns
        -------
        Dict[str, Any]
            Standardized prediction output containing predicted attack, confidence,
            and class probabilities.
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

        # Validate schema and enforce column ordering
        df_val, _ = validate_feature_data(df_in, expected_features=self.feature_names)

        probs = self.model.predict_proba(df_val)[0]
        pred_idx = int(np.argmax(probs))
        pred_label = str(self.classes[pred_idx])
        confidence = float(probs[pred_idx])

        class_probabilities: Dict[str, float] = {
            str(cls_name): float(prob) for cls_name, prob in zip(self.classes, probs)
        }

        return {
            "model_name": self.model_name,
            "predicted_attack": pred_label,
            "confidence": confidence,
            "class_probabilities": class_probabilities,
        }

    def predict_batch(self, df_features: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate predictions for a batch of network flow vectors."""
        df_val, _ = validate_feature_data(df_features, expected_features=self.feature_names)
        all_probs = self.model.predict_proba(df_val)
        results = []

        for probs in all_probs:
            pred_idx = int(np.argmax(probs))
            results.append({
                "predicted_attack": str(self.classes[pred_idx]),
                "confidence": float(probs[pred_idx]),
                "class_probabilities": {
                    str(cls_name): float(p) for cls_name, p in zip(self.classes, probs)
                },
            })
        return results


def predict_attack_class(
    feature_vector: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    model_path: Union[str, Path] = "ml/models/random_forest_classifier.joblib",
) -> Dict[str, Any]:
    """Convenience functional interface for attack prediction."""
    predictor = AttackClassifierPredictor(model_path)
    return predictor.predict_single(feature_vector)
