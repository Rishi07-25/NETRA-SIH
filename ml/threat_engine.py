"""Unified Threat Assessment Engine for NETRA Stage 2.

Combines outputs from supervised attack classification and unsupervised anomaly detection
into a transparent, centralized threat assessment with deterministic heuristic severity levels.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ml.anomaly_detection.predict import AnomalyDetectorPredictor
from ml.attack_classification.predict import AttackClassifierPredictor

logger = logging.getLogger(__name__)

# Configurable threat level mapping thresholds (MVP Heuristics)
# NOTE: These levels represent operational decision heuristics, NOT calibrated probabilities.
DEFAULT_THREAT_CONFIG: Dict[str, Any] = {
    # High threat threshold: attack predicted with high confidence OR combined anomaly
    "high_confidence_threshold": 0.80,
    "medium_confidence_threshold": 0.40,
    "anomaly_display_threshold": 0.65,
}


class ThreatEngine:
    """Unified threat engine combining supervised classification and anomaly telemetry."""

    def __init__(
        self,
        classifier_model: Union[str, Path, AttackClassifierPredictor] = "ml/models/random_forest_classifier.joblib",
        anomaly_model: Union[str, Path, AnomalyDetectorPredictor] = "ml/models/isolation_forest.joblib",
        threat_config: Optional[Dict[str, Any]] = None,
    ):
        if isinstance(classifier_model, AttackClassifierPredictor):
            self.classifier = classifier_model
        else:
            self.classifier = AttackClassifierPredictor(classifier_model)

        if isinstance(anomaly_model, AnomalyDetectorPredictor):
            self.anomaly_detector = anomaly_model
        else:
            self.anomaly_detector = AnomalyDetectorPredictor(anomaly_model)

        self.config = threat_config or DEFAULT_THREAT_CONFIG

    def determine_threat_level(
        self,
        predicted_attack: str,
        confidence: float,
        is_anomalous: bool,
        anomaly_display_score: float,
    ) -> str:
        """Evaluate deterministic heuristic rules to assign a categorical threat level.

        Decision Rules:
        - HIGH:
          Non-benign attack predicted with >= high_confidence (e.g. 0.80) OR
          Non-benign attack with medium confidence AND anomalous flag triggered OR
          Severe anomaly score (>= anomaly_display_threshold).
        - MEDIUM:
          Non-benign attack predicted with >= medium_confidence (0.40) OR
          Benign classification but marked anomalous by baseline detector.
        - LOW:
          Benign classification with low anomaly score and no anomaly flag.

        Returns
        -------
        str: 'LOW', 'MEDIUM', or 'HIGH'
        """
        is_attack = (predicted_attack.upper() != "BENIGN")

        high_conf = self.config.get("high_confidence_threshold", 0.80)
        med_conf = self.config.get("medium_confidence_threshold", 0.40)
        anom_thresh = self.config.get("anomaly_display_threshold", 0.65)

        if is_attack and confidence >= high_conf:
            return "HIGH"
        if is_attack and is_anomalous:
            return "HIGH"
        if anomaly_display_score >= anom_thresh:
            return "HIGH"

        if is_attack and confidence >= med_conf:
            return "MEDIUM"
        if (not is_attack) and is_anomalous:
            return "MEDIUM"

        return "LOW"

    def assess_threat(
        self,
        feature_vector: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    ) -> Dict[str, Any]:
        """Generate a unified threat assessment for incoming network telemetry.

        Parameters
        ----------
        feature_vector : Dict, pd.Series, or 1-row DataFrame
            Network flow features.

        Returns
        -------
        Dict[str, Any]
            Unified result combining classification, anomaly metrics, and threat level.
        """
        cls_res = self.classifier.predict_single(feature_vector)
        anom_res = self.anomaly_detector.predict_single(feature_vector)

        threat_level = self.determine_threat_level(
            predicted_attack=cls_res["predicted_attack"],
            confidence=cls_res["confidence"],
            is_anomalous=anom_res["is_anomalous"],
            anomaly_display_score=anom_res["display_score"],
        )

        return {
            "predicted_attack": cls_res["predicted_attack"],
            "classification_confidence": cls_res["confidence"],
            "class_probabilities": cls_res["class_probabilities"],
            "is_anomalous": anom_res["is_anomalous"],
            "anomaly_score": anom_res["display_score"],
            "raw_anomaly_score": anom_res["raw_score"],
            "threat_level": threat_level,
            "threat_level_heuristic_note": (
                "Threat level is an operational decision heuristic based on classifier confidence "
                "and anomaly evidence. It is not a calibrated statistical probability."
            ),
        }


def evaluate_threat(
    feature_vector: Union[Dict[str, Any], pd.Series, pd.DataFrame],
    classifier_path: Union[str, Path] = "ml/models/random_forest_classifier.joblib",
    anomaly_path: Union[str, Path] = "ml/models/isolation_forest.joblib",
) -> Dict[str, Any]:
    """Convenience functional interface for unified threat assessment."""
    engine = ThreatEngine(classifier_model=classifier_path, anomaly_model=anomaly_path)
    return engine.assess_threat(feature_vector)
