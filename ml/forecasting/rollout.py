"""Autoregressive Multi-Step State Rollout Engine for NETRA Stage 3B.2.

Provides:
1. Recursive state simulation:
   [S_(t-L+1), ..., S_t] -> predict S_hat_(t+1) -> shift -> predict S_hat_(t+2) -> ... -> S_hat_(t+K)
2. Decoupled multi-horizon threat projection (attack probability, risk score, type, stage).
3. Persistence baseline for state forecasting (S_hat_(t+k) = S_t).
4. Horizon degradation analysis (evaluates metrics across horizons K=1, 2, 3, 5).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
import torch

from ml.forecasting.schema import TAXONOMY_DISCLAIMER
from ml.forecasting.world_model import TemporalWorldModel

logger = logging.getLogger(__name__)


def validate_rollout_input(
    X: Union[np.ndarray, torch.Tensor],
    expected_dim: int = 19,
    expected_len: int = 5,
) -> torch.Tensor:
    """Validate input sequence tensor for rollout safety.

    Parameters
    ----------
    X : np.ndarray or torch.Tensor
        Input sequence tensor of shape (N, L, D).
    expected_dim : int
        Expected feature dimension D (default 19).
    expected_len : int
        Expected observation length L (default 5).

    Returns
    -------
    torch.Tensor
        Validated float32 PyTorch tensor.
    """
    if isinstance(X, np.ndarray):
        if not np.isfinite(X).all():
            raise ValueError("Input array contains NaN or Infinite values.")
        tensor = torch.from_numpy(X.astype(np.float32))
    elif isinstance(X, torch.Tensor):
        if not torch.isfinite(X).all():
            raise ValueError("Input tensor contains NaN or Infinite values.")
        tensor = X.float()
    else:
        raise TypeError(f"Expected numpy.ndarray or torch.Tensor, got {type(X)}")

    if tensor.dim() != 3:
        raise ValueError(f"Expected 3D tensor (N, {expected_len}, {expected_dim}), got shape {tensor.shape}")

    if tensor.size(1) != expected_len:
        raise ValueError(f"Expected sequence length L={expected_len}, got {tensor.size(1)}")

    if tensor.size(2) != expected_dim:
        raise ValueError(f"Feature dimension mismatch: expected {expected_dim}, got {tensor.size(2)}")

    return tensor


def persistence_baseline(
    initial_sequence: Union[np.ndarray, torch.Tensor],
    steps: int = 5,
) -> np.ndarray:
    """Predict future states using naive persistence baseline: S_hat_(t+k) = S_t.

    Parameters
    ----------
    initial_sequence : np.ndarray or torch.Tensor
        Initial observation window of shape (N, L, D).
    steps : int
        Number of forward forecast steps K >= 1.

    Returns
    -------
    np.ndarray
        Baseline predictions of shape (N, K, D) repeating S_t for all k in [1, K].
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")

    tensor = validate_rollout_input(initial_sequence)
    # Extract S_t (last observed step in sequence): shape (N, D)
    last_state = tensor[:, -1, :].cpu().numpy()  # (N, D)
    N, D = last_state.shape

    # Tile across K steps: (N, K, D)
    persistence_states = np.repeat(last_state[:, np.newaxis, :], steps, axis=1)
    return persistence_states


class StateRolloutEngine:
    """Autoregressive simulation engine for temporal network world models."""

    def __init__(
        self,
        model: TemporalWorldModel,
        type_to_idx: Dict[str, int],
        stage_to_idx: Dict[str, int],
        scaler: Optional[Any] = None,
        device: Optional[str] = None,
    ):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.model.eval()

        self.type_to_idx = type_to_idx
        self.idx_to_type = {v: k for k, v in type_to_idx.items()}

        self.stage_to_idx = stage_to_idx
        self.idx_to_stage = {v: k for k, v in stage_to_idx.items()}

        self.scaler = scaler

    def rollout(
        self,
        initial_sequence: Union[np.ndarray, torch.Tensor],
        steps: int = 5,
        scale_input: bool = False,
        return_inverse_scaled: bool = False,
    ) -> Dict[str, Any]:
        """Perform recursive autoregressive rollout for K forward steps.

        Algorithm:
        For k = 1 to K:
          1. Predict: S_hat_(t+k) and threats from current sequence [S_(t+k-L), ..., S_(t+k-1)]
          2. Store predictions
          3. Shift sequence: drop oldest state, append S_hat_(t+k)
          4. Repeat without teacher forcing or ground-truth injection

        Parameters
        ----------
        initial_sequence : np.ndarray or torch.Tensor
            Observation sequence of shape (N, L, D).
        steps : int
            Forecast horizon steps K >= 1.
        scale_input : bool
            If True, applies scaler.transform to initial_sequence before rollout.
        return_inverse_scaled : bool
            If True and scaler is present, returns inverse-scaled physical state predictions.

        Returns
        -------
        Dict[str, Any]
            Rollout trajectory bundle:
            - "predicted_states": np.ndarray (N, K, D)
            - "predicted_states_physical": Optional[np.ndarray] (N, K, D) if inverse scaled
            - "future_attack_probability": np.ndarray (N, K) in [0.0, 1.0]
            - "predicted_attack_binary": np.ndarray (N, K) in {0, 1}
            - "predicted_risk_score": np.ndarray (N, K) in [0.0, 1.0]
            - "predicted_attack_type": List[List[str]] of shape (N, K)
            - "attack_type_confidence": np.ndarray (N, K) in [0.0, 1.0]
            - "predicted_attack_stage": List[List[str]] of shape (N, K)
            - "stage_confidence": np.ndarray (N, K) in [0.0, 1.0]
            - "steps": int
            - "taxonomy_disclaimer": str
        """
        if steps < 1:
            raise ValueError(f"Rollout steps K must be >= 1, got {steps}")

        if isinstance(initial_sequence, np.ndarray) and scale_input:
            if self.scaler is None:
                raise ValueError("scale_input=True requested but no scaler is registered.")
            N, L, D = initial_sequence.shape
            init_flat = self.scaler.transform(initial_sequence.reshape(-1, D))
            initial_sequence = init_flat.reshape(N, L, D).astype(np.float32)

        current_seq = validate_rollout_input(initial_sequence).to(self.device)
        N, L, D = current_seq.shape

        rollout_states = []
        rollout_attack_prob = []
        rollout_binary = []
        rollout_risk = []
        rollout_types = []
        rollout_type_conf = []
        rollout_stages = []
        rollout_stage_conf = []

        self.model.eval()
        with torch.no_grad():
            for step in range(steps):
                # Forward pass on current sliding sequence
                outputs = self.model(current_seq)

                # 1. State prediction S_hat_(t+k)
                next_state = outputs["next_state"]  # shape (N, D)
                if not torch.isfinite(next_state).all():
                    raise RuntimeError(f"Non-finite output detected in predicted state at rollout step {step + 1}")

                rollout_states.append(next_state.cpu().numpy())

                # 2. Binary Attack Probability
                binary_probs = torch.softmax(outputs["binary_logits"], dim=-1)
                attack_prob = binary_probs[:, 1].cpu().numpy()
                pred_binary = torch.argmax(binary_probs, dim=-1).cpu().numpy()
                rollout_attack_prob.append(attack_prob)
                rollout_binary.append(pred_binary)

                # 3. Continuous Risk Score
                risk_score = outputs["risk_score"].cpu().numpy()
                rollout_risk.append(risk_score)

                # 4. Attack Type & Confidence
                type_probs = torch.softmax(outputs["type_logits"], dim=-1)
                t_conf, t_idx = torch.max(type_probs, dim=-1)
                step_types = [self.idx_to_type.get(int(i), "Unknown") for i in t_idx.cpu().numpy()]
                rollout_types.append(step_types)
                rollout_type_conf.append(t_conf.cpu().numpy())

                # 5. Attack Stage & Confidence
                stage_probs = torch.softmax(outputs["stage_logits"], dim=-1)
                s_conf, s_idx = torch.max(stage_probs, dim=-1)
                step_stages = [self.idx_to_stage.get(int(i), "Normal") for i in s_idx.cpu().numpy()]
                rollout_stages.append(step_stages)
                rollout_stage_conf.append(s_conf.cpu().numpy())

                # 6. Autoregressive Shift: Drop oldest, append S_hat_(t+k)
                # current_seq[:, 1:, :] shape: (N, L-1, D)
                # next_state.unsqueeze(1) shape: (N, 1, D)
                current_seq = torch.cat([current_seq[:, 1:, :], next_state.unsqueeze(1)], dim=1)

        # Transpose arrays from (K, N, ...) to (N, K, ...)
        pred_states_arr = np.transpose(np.array(rollout_states), (1, 0, 2))  # (N, K, D)
        attack_prob_arr = np.transpose(np.array(rollout_attack_prob), (1, 0))  # (N, K)
        binary_arr = np.transpose(np.array(rollout_binary), (1, 0))  # (N, K)
        risk_arr = np.transpose(np.array(rollout_risk), (1, 0))  # (N, K)
        type_conf_arr = np.transpose(np.array(rollout_type_conf), (1, 0))  # (N, K)
        stage_conf_arr = np.transpose(np.array(rollout_stage_conf), (1, 0))  # (N, K)

        # Types and stages to list of length N with K elements each
        types_by_sample = [[rollout_types[k][n] for k in range(steps)] for n in range(N)]
        stages_by_sample = [[rollout_stages[k][n] for k in range(steps)] for n in range(N)]

        result: Dict[str, Any] = {
            "predicted_states": pred_states_arr,
            "future_attack_probability": np.round(attack_prob_arr.astype(float), 4),
            "predicted_attack_binary": binary_arr.astype(int),
            "predicted_risk_score": np.round(risk_arr.astype(float), 4),
            "predicted_attack_type": types_by_sample,
            "attack_type_confidence": np.round(type_conf_arr.astype(float), 4),
            "predicted_attack_stage": stages_by_sample,
            "stage_confidence": np.round(stage_conf_arr.astype(float), 4),
            "steps": steps,
            "taxonomy_disclaimer": TAXONOMY_DISCLAIMER,
        }

        if return_inverse_scaled and self.scaler is not None:
            flat_states = pred_states_arr.reshape(-1, D)
            inv_states = self.scaler.inverse_transform(flat_states).reshape(N, steps, D)
            result["predicted_states_physical"] = inv_states

        return result


def evaluate_horizon_degradation(
    rollout_engine: StateRolloutEngine,
    X_test: np.ndarray,
    ground_truth_trajectories: np.ndarray,
    ground_truth_threats: Optional[Dict[str, Any]] = None,
    horizons: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Evaluate predictive degradation across multi-step forecast horizons.

    Compares LSTM rollout against ground-truth trajectory and naive persistence baseline.

    Parameters
    ----------
    rollout_engine : StateRolloutEngine
        Fitted rollout engine.
    X_test : np.ndarray
        Input test sequences of shape (N, L, D).
    ground_truth_trajectories : np.ndarray
        True future states of shape (N, K_max, D) standardized with the same scaler.
    ground_truth_threats : Optional[Dict[str, Any]]
        Optional ground truth threat targets per horizon step:
        {"binary": (N, K_max), "risk": (N, K_max), "type": (N, K_max), "stage": (N, K_max)}
    horizons : Optional[List[int]]
        List of horizon steps to evaluate (e.g. [1, 2, 3, 5]).

    Returns
    -------
    Dict[str, Any]
        Horizon metrics dictionary and degradation summary table.
    """
    N, K_max, D = ground_truth_trajectories.shape
    eval_horizons = [h for h in (horizons or [1, 2, 3, 5]) if h <= K_max]

    if not eval_horizons:
        return {"status": "NO_VALID_HORIZONS", "horizon_table": []}

    # Execute model rollout for K_max steps
    rollout_results = rollout_engine.rollout(X_test, steps=K_max)
    pred_states = rollout_results["predicted_states"]  # (N, K_max, D)

    # Persistence baseline
    persist_states = persistence_baseline(X_test, steps=K_max)  # (N, K_max, D)

    horizon_metrics = []

    for h in eval_horizons:
        h_idx = h - 1  # 0-indexed step

        # 1. State Prediction Error (LSTM vs Persistence)
        true_h = ground_truth_trajectories[:, h_idx, :]
        pred_h = pred_states[:, h_idx, :]
        pers_h = persist_states[:, h_idx, :]

        state_mae = float(mean_absolute_error(true_h, pred_h))
        state_rmse = float(np.sqrt(mean_squared_error(true_h, pred_h)))

        persist_mae = float(mean_absolute_error(true_h, pers_h))
        persist_rmse = float(np.sqrt(mean_squared_error(true_h, pers_h)))

        # 2. Threat Metrics if available
        attack_f1 = None
        risk_mae = None
        type_f1 = None
        stage_f1 = None

        if ground_truth_threats is not None:
            if "binary" in ground_truth_threats:
                true_bin = ground_truth_threats["binary"][:, h_idx]
                pred_bin = rollout_results["predicted_attack_binary"][:, h_idx]
                attack_f1 = float(f1_score(true_bin, pred_bin, zero_division=0))

            if "risk" in ground_truth_threats:
                true_risk = ground_truth_threats["risk"][:, h_idx]
                pred_risk = rollout_results["predicted_risk_score"][:, h_idx]
                risk_mae = float(mean_absolute_error(true_risk, pred_risk))

            if "type" in ground_truth_threats:
                true_type = ground_truth_threats["type"][:, h_idx]
                pred_type = [rollout_results["predicted_attack_type"][n][h_idx] for n in range(N)]
                type_f1 = float(f1_score(true_type, pred_type, average="macro", zero_division=0))

            if "stage" in ground_truth_threats:
                true_stage = ground_truth_threats["stage"][:, h_idx]
                pred_stage = [rollout_results["predicted_attack_stage"][n][h_idx] for n in range(N)]
                stage_f1 = float(f1_score(true_stage, pred_stage, average="macro", zero_division=0))

        horizon_metrics.append({
            "horizon": h,
            "state_mae": round(state_mae, 4),
            "state_rmse": round(state_rmse, 4),
            "persistence_mae": round(persist_mae, 4),
            "persistence_rmse": round(persist_rmse, 4),
            "attack_f1": round(attack_f1, 4) if attack_f1 is not None else None,
            "risk_mae": round(risk_mae, 4) if risk_mae is not None else None,
            "type_f1": round(type_f1, 4) if type_f1 is not None else None,
            "stage_f1": round(stage_f1, 4) if stage_f1 is not None else None,
        })

    return {
        "num_samples": N,
        "max_horizon": K_max,
        "horizons_evaluated": eval_horizons,
        "horizon_metrics": horizon_metrics,
    }
