"""Prediction and inference interface for NETRA Stage 3B.1 Temporal World Model.

Accepts sequence tensors of shape (N, L, D) and returns structured predictions
disentangling future attack probability from future risk score.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from ml.forecasting.schema import TAXONOMY_DISCLAIMER
from ml.forecasting.world_model import CANONICAL_ATTACK_TYPES, TemporalWorldModel

logger = logging.getLogger(__name__)


class WorldModelPredictor:
    """Inference engine for the trained Temporal World Model."""

    def __init__(
        self,
        model: TemporalWorldModel,
        type_to_idx: Dict[str, int],
        stage_to_idx: Dict[str, int],
        scaler: Optional[Any] = None,
        device: Optional[str] = None,
        feature_names: Optional[List[str]] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.type_to_idx = type_to_idx
        self.idx_to_type = {v: k for k, v in type_to_idx.items()}

        self.stage_to_idx = stage_to_idx
        self.idx_to_stage = {v: k for k, v in stage_to_idx.items()}

        self.scaler = scaler
        self.feature_names = feature_names or getattr(model, "feature_names", None)

    @classmethod
    def load_from_checkpoint(
        cls,
        checkpoint_path: Union[str, Path],
        device: Optional[str] = None,
    ) -> WorldModelPredictor:
        """Instantiate predictor directly from a saved checkpoint bundle."""
        p = Path(checkpoint_path)
        if not p.exists():
            raise FileNotFoundError(f"World model checkpoint not found: {p}")

        try:
            ckpt = torch.load(p, map_location="cpu", weights_only=False)
        except TypeError:
            # For older torch versions where weights_only is not an argument
            ckpt = torch.load(p, map_location="cpu")

        arch = ckpt["architecture"]
        model = TemporalWorldModel(
            input_size=arch["input_size"],
            hidden_size=arch["hidden_size"],
            num_layers=arch["num_layers"],
            dropout=arch.get("dropout", 0.0),
            attack_types=ckpt["attack_types"],
            attack_stages=ckpt["attack_stages"],
            feature_names=ckpt.get("feature_names"),
        )
        model.load_state_dict(ckpt["model_state_dict"])

        return cls(
            model=model,
            type_to_idx=ckpt["type_to_idx"],
            stage_to_idx=ckpt["stage_to_idx"],
            scaler=ckpt.get("scaler"),
            device=device,
            feature_names=ckpt.get("feature_names"),
        )

    def predict(
        self,
        X: Union[np.ndarray, torch.Tensor],
        scale_input: bool = False,
    ) -> Dict[str, Any]:
        """Perform forward inference on a batch of temporal sequences.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Sequence array of shape (N, L, D) where L=5, D=19.
        scale_input : bool
            If True and a fitted scaler is registered, applies scaler.transform
            to X before inference.

        Returns
        -------
        Dict[str, Any]
            Structured prediction dictionary:
            - "future_attack_probability": np.ndarray (N,) in [0.0, 1.0] (binary attack probability)
            - "predicted_attack_binary": np.ndarray (N,) in {0, 1}
            - "predicted_attack_type": List[str] length N
            - "attack_type_confidence": np.ndarray (N,) in [0.0, 1.0]
            - "predicted_attack_stage": List[str] length N
            - "stage_confidence": np.ndarray (N,) in [0.0, 1.0]
            - "predicted_risk_score": np.ndarray (N,) in [0.0, 1.0] (attack flow density)
            - "taxonomy_disclaimer": str
        """
        if isinstance(X, np.ndarray):
            X_arr = X.copy()
            if scale_input:
                if self.scaler is None:
                    raise ValueError("scale_input=True requested but no scaler is registered.")
                N, L, D = X_arr.shape
                X_arr = self.scaler.transform(X_arr.reshape(-1, D)).reshape(N, L, D)
            x_tensor = torch.from_numpy(X_arr.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor):
            x_tensor = X.float().to(self.device)
        else:
            raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(X)}")

        if x_tensor.dim() != 3:
            raise ValueError(f"Expected 3D input (N, L, D), got shape {x_tensor.shape}")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)

            # 1. Binary Attack Probability & Prediction
            binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)
            future_attack_prob = binary_probs[:, 1].cpu().numpy()  # P(Attack)
            pred_binary = torch.argmax(binary_probs, dim=-1).cpu().numpy()

            # 2. Attack Type
            type_probs = torch.softmax(outputs["type_logits"], dim=-1)
            type_conf, type_indices = torch.max(type_probs, dim=-1)
            pred_type = [self.idx_to_type.get(idx, "Unknown") for idx in type_indices.cpu().numpy()]
            type_conf_arr = type_conf.cpu().numpy()

            # 3. Operational Attack Stage
            stage_probs = torch.softmax(outputs["stage_logits"], dim=-1)
            stage_conf, stage_indices = torch.max(stage_probs, dim=-1)
            pred_stage = [self.idx_to_stage.get(idx, "Normal") for idx in stage_indices.cpu().numpy()]
            stage_conf_arr = stage_conf.cpu().numpy()

            # 4. Continuous Risk Score (Empirical Attack Flow Fraction)
            predicted_risk = outputs["risk_score"].cpu().numpy()

            # 5. Standardized Next-State Prediction (19 features)
            predicted_state = (
                outputs["next_state"].cpu().numpy()
                if "next_state" in outputs
                else None
            )

        res: Dict[str, Any] = {
            "future_attack_probability": np.round(future_attack_prob.astype(float), 4),
            "predicted_attack_binary": pred_binary.astype(int),
            "predicted_attack_type": pred_type,
            "attack_type_confidence": np.round(type_conf_arr.astype(float), 4),
            "predicted_attack_stage": pred_stage,
            "stage_confidence": np.round(stage_conf_arr.astype(float), 4),
            "predicted_risk_score": np.round(predicted_risk.astype(float), 4),
            "taxonomy_disclaimer": TAXONOMY_DISCLAIMER,
        }
        if predicted_state is not None:
            res["predicted_next_state"] = predicted_state.astype(np.float32)
        return res

    def inverse_scale_state(
        self,
        standardized_state: np.ndarray,
    ) -> np.ndarray:
        """Convert standardized 19D state vector back to physical network units.

        Parameters
        ----------
        standardized_state : np.ndarray
            Array of shape (N, 19) or (N, K, 19) containing standardized values.

        Returns
        -------
        np.ndarray
            Array of same shape in physical feature units.
        """
        if self.scaler is None:
            raise ValueError("No scaler registered in WorldModelPredictor. Cannot inverse scale.")

        orig_shape = standardized_state.shape
        D = orig_shape[-1]
        flat_arr = standardized_state.reshape(-1, D)
        inv_flat = self.scaler.inverse_transform(flat_arr)
        return inv_flat.reshape(orig_shape)

    def rollout(
        self,
        initial_sequence: Union[np.ndarray, torch.Tensor],
        steps: int = 5,
        scale_input: bool = False,
        return_inverse_scaled: bool = False,
    ) -> Dict[str, Any]:
        """Perform autoregressive multi-step rollout for K forward horizons."""
        from ml.forecasting.rollout import StateRolloutEngine

        engine = StateRolloutEngine(
            model=self.model,
            type_to_idx=self.type_to_idx,
            stage_to_idx=self.stage_to_idx,
            scaler=self.scaler,
            device=str(self.device),
        )
        return engine.rollout(
            initial_sequence=initial_sequence,
            steps=steps,
            scale_input=scale_input,
            return_inverse_scaled=return_inverse_scaled,
        )

    def get_latent_embedding(
        self,
        X: Union[np.ndarray, torch.Tensor],
        scale_input: bool = False,
    ) -> np.ndarray:
        """Expose 64-dimensional latent temporal embedding for analysis/interpretability.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input sequence tensor of shape (N, L, D).

        Returns
        -------
        np.ndarray
            Latent temporal representation of shape (N, 64).
        """
        if isinstance(X, np.ndarray):
            X_arr = X.copy()
            if scale_input:
                if self.scaler is None:
                    raise ValueError("scale_input=True requested but no scaler is registered.")
                N, L, D = X_arr.shape
                X_arr = self.scaler.transform(X_arr.reshape(-1, D)).reshape(N, L, D)
            x_tensor = torch.from_numpy(X_arr.astype(np.float32)).to(self.device)
        elif isinstance(X, torch.Tensor):
            x_tensor = X.float().to(self.device)
        else:
            raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(X)}")

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(x_tensor)
            emb = outputs["embedding"].cpu().numpy()
        return emb

    def predict_dataframe(
        self,
        X: np.ndarray,
        scale_input: bool = False,
    ) -> pd.DataFrame:
        """Return structured inference outputs as a pandas DataFrame."""
        preds = self.predict(X, scale_input=scale_input)
        df_dict = {
            "future_attack_binary": preds["predicted_attack_binary"],
            "future_attack_probability": preds["future_attack_probability"],
            "predicted_attack_type": preds["predicted_attack_type"],
            "attack_type_confidence": preds["attack_type_confidence"],
            "predicted_attack_stage": preds["predicted_attack_stage"],
            "stage_confidence": preds["stage_confidence"],
            "predicted_risk_score": preds["predicted_risk_score"],
        }
        return pd.DataFrame(df_dict)
