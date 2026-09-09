"""Service managing attack classification and unified threat assessment inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ml.attack_classification.predict import AttackClassifierPredictor
from ml.threat_engine import ThreatEngine

logger = logging.getLogger(__name__)

DEFAULT_CLASSIFIER_PATH = "ml/models/random_forest_classifier.joblib"
DEFAULT_ANOMALY_PATH = "ml/models/isolation_forest.joblib"


class PredictionService:
    """Service orchestrator for threat classification and unified assessment."""

    def __init__(
        self,
        classifier_model_path: Optional[Union[str, Path]] = None,
        anomaly_model_path: Optional[Union[str, Path]] = None,
    ):
        self.classifier_path = Path(classifier_model_path or DEFAULT_CLASSIFIER_PATH)
        self.anomaly_path = Path(anomaly_model_path or DEFAULT_ANOMALY_PATH)

        self._predictor: Optional[AttackClassifierPredictor] = None
        self._threat_engine: Optional[ThreatEngine] = None

    @property
    def predictor(self) -> AttackClassifierPredictor:
        if self._predictor is None:
            if not self.classifier_path.exists():
                raise FileNotFoundError(
                    f"Attack classifier model not found at {self.classifier_path}. "
                    "Run Stage 2 training or ensure models exist."
                )
            self._predictor = AttackClassifierPredictor(self.classifier_path)
        return self._predictor

    @property
    def threat_engine(self) -> ThreatEngine:
        if self._threat_engine is None:
            if not self.classifier_path.exists() or not self.anomaly_path.exists():
                raise FileNotFoundError(
                    f"Model artifacts missing: classifier={self.classifier_path.exists()}, "
                    f"anomaly={self.anomaly_path.exists()}"
                )
            self._threat_engine = ThreatEngine(
                classifier_model=self.classifier_path,
                anomaly_model=self.anomaly_path,
            )
        return self._threat_engine

    def classify_threat(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Execute model classification over input features."""
        pred = self.predictor.predict_single(features)
        
        # Calculate top contributing features if Random Forest model feature importances exist
        top_features: List[str] = []
        rf_model = getattr(self.predictor.model, "feature_importances_", None)
        if rf_model is not None:
            sorted_indices = rf_model.argsort()[::-1][:5]
            top_features = [self.predictor.feature_names[i] for i in sorted_indices]

        return {
            "attack_class": pred["predicted_attack"],
            "confidence": pred["confidence"],
            "class_probabilities": pred["class_probabilities"],
            "top_features": top_features,
            "model_name": pred.get("model_name", "RandomForestClassifier"),
        }

    def assess_full_threat(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Execute unified threat assessment combining classification, anomaly, and threat level."""
        return self.threat_engine.assess_threat(features)
