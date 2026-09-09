"""Service managing temporal attack risk forecasting and multi-step trajectory projection."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from ml.forecasting.predict_world_model import WorldModelPredictor
from ml.forecasting.rollout import StateRolloutEngine
from ml.forecasting.schema import STATE_PREDICTIVE_FEATURES, TAXONOMY_DISCLAIMER

logger = logging.getLogger(__name__)

DEFAULT_MODEL_CHECKPOINT = "ml/models/temporal_world_model_v2.pt"


class ForecastingService:
    """Service orchestrator for temporal risk projection and multi-horizon forecasting."""

    def __init__(self, model_checkpoint_path: Optional[Union[str, Path]] = None):
        self.checkpoint_path = Path(model_checkpoint_path or DEFAULT_MODEL_CHECKPOINT)
        self._predictor: Optional[WorldModelPredictor] = None
        self._rollout_engine: Optional[StateRolloutEngine] = None

    @property
    def predictor(self) -> WorldModelPredictor:
        if self._predictor is None:
            if not self.checkpoint_path.exists():
                raise FileNotFoundError(
                    f"Temporal world model checkpoint not found at {self.checkpoint_path}. "
                    "Ensure Stage 3B model is trained or checkpoint is provided."
                )
            self._predictor = WorldModelPredictor.load_from_checkpoint(self.checkpoint_path)
        return self._predictor

    @property
    def rollout_engine(self) -> StateRolloutEngine:
        if self._rollout_engine is None:
            pred = self.predictor
            self._rollout_engine = StateRolloutEngine(
                model=pred.model,
                type_to_idx=pred.type_to_idx,
                stage_to_idx=pred.stage_to_idx,
                scaler=pred.scaler,
                device=str(pred.device),
            )
        return self._rollout_engine

    def _convert_history_to_matrix(
        self, historical_windows: List[Dict[str, float]], expected_len: int = 5
    ) -> np.ndarray:
        """Convert a list of feature dictionaries into an (1, L, D) matrix."""
        feature_names = getattr(self.predictor, "feature_names", None) or STATE_PREDICTIVE_FEATURES
        D = len(feature_names)
        
        if len(historical_windows) == 0:
            raise ValueError("Historical windows list cannot be empty.")

        # Extract rows
        rows = []
        for win in historical_windows:
            row = [float(win.get(feat, 0.0)) for feat in feature_names]
            rows.append(row)

        arr = np.array(rows, dtype=np.float32)
        if len(arr) < expected_len:
            # Pad beginning with repetition of the earliest state if shorter
            pad_count = expected_len - len(arr)
            padding = np.repeat(arr[:1, :], pad_count, axis=0)
            arr = np.vstack([padding, arr])
        elif len(arr) > expected_len:
            # Take the most recent expected_len windows
            arr = arr[-expected_len:, :]

        return arr.reshape(1, expected_len, D)

    def generate_risk_forecast(self, historical_windows: List[Dict[str, float]]) -> Dict[str, Any]:
        """Generate forward-looking risk probabilities across multiple time horizons."""
        expected_len = getattr(self.predictor.model, "num_layers", 5)
        # Sequence length L is typically 5 for the temporal world model
        seq_len = 5
        input_tensor = self._convert_history_to_matrix(historical_windows, expected_len=seq_len)

        # Execute 3-step rollout corresponding to horizons 1m, 5m, 15m (or steps k=1, 2, 3)
        horizons_meta = ["1m", "5m", "15m"]
        rollout_res = self.rollout_engine.rollout(
            initial_sequence=input_tensor,
            steps=3,
            scale_input=True if self.predictor.scaler is not None else False,
        )

        probs = rollout_res["future_attack_probability"][0]  # shape (3,)
        risks = rollout_res["predicted_risk_score"][0]        # shape (3,)
        pred_types = rollout_res["predicted_attack_type"][0]  # shape (3,)
        pred_stages = rollout_res["predicted_attack_stage"][0]# shape (3,)

        forecasts = []
        for idx, h_name in enumerate(horizons_meta):
            p = float(probs[idx]) if idx < len(probs) else float(probs[-1])
            r = float(risks[idx]) * 100.0 if idx < len(risks) else float(risks[-1]) * 100.0
            forecasts.append({
                "horizon": h_name,
                "risk_score": round(r, 2),
                "attack_probability": round(p, 4),
                "predicted_type": pred_types[idx] if idx < len(pred_types) else pred_types[-1],
                "predicted_stage": pred_stages[idx] if idx < len(pred_stages) else pred_stages[-1],
            })

        # Calculate trajectory trend
        p0 = float(probs[0])
        p_last = float(probs[-1])
        if p_last - p0 > 0.05:
            trajectory = "ESCALATING"
        elif p0 - p_last > 0.05:
            trajectory = "DECLINING"
        else:
            trajectory = "STABLE"

        # Current threat score baseline
        current_threat_score = round(float(probs[0]) * 100.0, 2)

        return {
            "current_threat_score": current_threat_score,
            "forecasts": forecasts,
            "trajectory": trajectory,
            "taxonomy_disclaimer": TAXONOMY_DISCLAIMER,
        }
