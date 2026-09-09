"""Service managing behavioral anomaly detection inference."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

from ml.anomaly_detection.predict import AnomalyDetectorPredictor

logger = logging.getLogger(__name__)

DEFAULT_ANOMALY_PATH = "ml/models/isolation_forest.joblib"


class AnomalyService:
    """Service orchestrator for behavioral anomaly scoring."""

    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        self.model_path = Path(model_path or DEFAULT_ANOMALY_PATH)
        self._predictor: Optional[AnomalyDetectorPredictor] = None

    @property
    def predictor(self) -> AnomalyDetectorPredictor:
        if self._predictor is None:
            if not self.model_path.exists():
                raise FileNotFoundError(
                    f"Anomaly model artifact not found at {self.model_path}. "
                    "Run Stage 2 training or ensure models exist."
                )
            self._predictor = AnomalyDetectorPredictor(self.model_path)
        return self._predictor

    def evaluate_anomaly(self, flow_sample: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate anomaly score for the given network flow sample."""
        result = self.predictor.predict_single(flow_sample)
        return {
            "is_anomalous": result["is_anomalous"],
            "anomaly_score": result["display_score"],
            "raw_score": result["raw_score"],
            "model_name": self.predictor.model_name,
        }
