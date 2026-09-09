"""Sliding sequence tensor builder for NETRA Stage 3A temporal forecasting.

Converts chronological sequence of network states S_1, ..., S_T into sliding
observation tensors X of shape (N, L, D) paired with temporal metadata.
Strictly ensures observation sequence ends before target window begins.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.forecasting.schema import STATE_PREDICTIVE_FEATURES

logger = logging.getLogger(__name__)


class SequenceBuilder:
    """Constructs fixed-length sliding temporal sequences from network state telemetry."""

    def __init__(
        self,
        sequence_length: int = 5,
        forecast_horizon: int = 1,
        feature_columns: Optional[List[str]] = None,
    ):
        """
        Parameters
        ----------
        sequence_length : int
            Number of historical state steps L in each sequence (default: 5).
        forecast_horizon : int
            Number of non-overlapping steps ahead H to project future threat state (default: 1).
        feature_columns : Optional[List[str]]
            Subset of predictive features to include. Defaults to STATE_PREDICTIVE_FEATURES.
        """
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if forecast_horizon < 1:
            raise ValueError(f"forecast_horizon must be >= 1, got {forecast_horizon}")

        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        self.feature_columns = feature_columns or STATE_PREDICTIVE_FEATURES

    def build_sequences(
        self,
        states_df: pd.DataFrame,
    ) -> Tuple[np.ndarray, pd.DataFrame, List[int]]:
        """Construct sequence array X of shape (N, L, D) and aligned metadata.

        Parameters
        ----------
        states_df : pd.DataFrame
            Chronological DataFrame containing state features and window metadata.

        Returns
        -------
        Tuple[np.ndarray, pd.DataFrame, List[int]]
            X_sequences : np.ndarray of shape (N, L, D)
            metadata_df : DataFrame of sequence observation & target timestamps
            target_indices : list of integer row indices in states_df corresponding to target window t + H
        """
        available_features = [c for c in self.feature_columns if c in states_df.columns]
        if not available_features:
            raise ValueError("None of the required predictive state features exist in states_df.")

        sort_col = "window_start" if "window_start" in states_df.columns else "window_id"
        sorted_states = states_df.sort_values(by=sort_col).reset_index(drop=True)

        T = len(sorted_states)
        L = self.sequence_length
        H = self.forecast_horizon

        if T < L:
            logger.warning(
                f"State count ({T}) is smaller than sequence_length ({L}). Cannot build sequences."
            )
            return np.empty((0, L, len(available_features)), dtype=np.float32), pd.DataFrame(), []

        feature_matrix = sorted_states[available_features].to_numpy(dtype=np.float32)

        sequences = []
        meta_records = []
        target_indices = []

        seq_id = 0
        has_time_cols = "window_start" in sorted_states.columns and "window_end" in sorted_states.columns

        for obs_end_idx in range(L - 1, T):
            obs_start_idx = obs_end_idx - L + 1

            if has_time_cols:
                obs_end_time = sorted_states.loc[obs_end_idx, "window_end"]
                # Find future windows whose window_start >= obs_end_time (strictly non-overlapping)
                candidate_targets = [
                    j for j in range(obs_end_idx + 1, T)
                    if sorted_states.loc[j, "window_start"] >= obs_end_time
                ]
                if len(candidate_targets) < H:
                    # Insufficient future non-overlapping windows
                    continue
                target_idx = candidate_targets[H - 1]
            else:
                target_idx = obs_end_idx + H
                if target_idx >= T:
                    continue

            # Slice window sequence [S_(t-L+1), ..., S_t]
            seq_x = feature_matrix[obs_start_idx : obs_end_idx + 1]  # shape (L, D)
            sequences.append(seq_x)
            target_indices.append(target_idx)

            obs_start_time = (
                sorted_states.loc[obs_start_idx, "window_start"] if has_time_cols else obs_start_idx
            )
            obs_end_time = (
                sorted_states.loc[obs_end_idx, "window_end"] if has_time_cols else obs_end_idx
            )
            target_start_time = (
                sorted_states.loc[target_idx, "window_start"] if has_time_cols else target_idx
            )
            target_end_time = (
                sorted_states.loc[target_idx, "window_end"] if has_time_cols else target_idx
            )

            meta_records.append({
                "sequence_id": seq_id,
                "obs_start_window": obs_start_idx,
                "obs_end_window": obs_end_idx,
                "target_window": target_idx,
                "observation_start": obs_start_time,
                "observation_end": obs_end_time,
                "target_start": target_start_time,
                "target_end": target_end_time,
                "sequence_length": L,
                "forecast_horizon": H,
            })
            seq_id += 1

        if not sequences:
            return np.empty((0, L, len(available_features)), dtype=np.float32), pd.DataFrame(), []

        X_array = np.array(sequences, dtype=np.float32)
        meta_df = pd.DataFrame(meta_records)

        logger.info(
            f"Constructed {len(X_array)} sequences of shape {X_array.shape} "
            f"(L={L}, H={H}, D={len(available_features)})."
        )
        return X_array, meta_df, target_indices


def create_sequences_from_states(
    states_df: pd.DataFrame,
    sequence_length: int = 5,
    forecast_horizon: int = 1,
) -> Tuple[np.ndarray, pd.DataFrame, List[int]]:
    """Functional convenience wrapper for sequence construction."""
    builder = SequenceBuilder(sequence_length=sequence_length, forecast_horizon=forecast_horizon)
    return builder.build_sequences(states_df)
