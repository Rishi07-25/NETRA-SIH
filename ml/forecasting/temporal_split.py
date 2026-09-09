"""Leakage-free temporal dataset partitioning and scaling for NETRA Stage 3A.

Implements partition-first chronological and scenario-aware splitting, generating
temporal sequences independently within each partition to mathematically prevent
cross-partition window leakage. Fits all feature scalers strictly on training data.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ml.forecasting.schema import STATE_PREDICTIVE_FEATURES
from ml.forecasting.sequence_builder import SequenceBuilder
from ml.forecasting.targets import extract_future_targets, validate_target_alignment

logger = logging.getLogger(__name__)


def partition_states_chronologically(
    states_df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Partition temporal states into chronological Train, Validation, and Test slices.

    Parameters
    ----------
    states_df : pd.DataFrame
        Chronologically sorted states DataFrame.
    train_ratio : float
        Proportion for training period (earliest).
    val_ratio : float
        Proportion for validation period (middle).
    test_ratio : float
        Proportion for test period (latest).

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        train_states, val_states, test_states
    """
    total = train_ratio + val_ratio + test_ratio
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")

    sort_col = "window_start" if "window_start" in states_df.columns else "window_id"
    sorted_df = states_df.sort_values(by=sort_col).reset_index(drop=True)
    n = len(sorted_df)

    if n == 0:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_states = sorted_df.iloc[:train_end].copy().reset_index(drop=True)
    val_states = sorted_df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_states = sorted_df.iloc[val_end:].copy().reset_index(drop=True)

    logger.info(
        f"Chronological split: Train={len(train_states)}, Val={len(val_states)}, Test={len(test_states)}."
    )
    return train_states, val_states, test_states


def partition_states_by_days(
    states_df: pd.DataFrame,
    train_days: Optional[List[str]] = None,
    val_days: Optional[List[str]] = None,
    test_days: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Scenario-aware partitioning by calendar days (e.g. CIC-IDS2017 Mon-Wed vs Thu vs Fri).

    If day lists are None, uses natural first 3 days for train, 4th for val, 5th for test.
    """
    if "window_start" not in states_df.columns:
        raise ValueError("states_df must contain 'window_start' timestamp column for day-based splitting.")

    df = states_df.copy()
    df["_date_str"] = pd.to_datetime(df["window_start"]).dt.strftime("%Y-%m-%d")
    unique_dates = sorted(list(df["_date_str"].unique()))

    if len(unique_dates) < 3:
        logger.warning(
            f"Dataset spans only {len(unique_dates)} distinct date(s); "
            "falling back to proportional chronological split."
        )
        return partition_states_chronologically(states_df)

    if train_days is None:
        # Default scenario-aware split (e.g. 5 days: 3 train, 1 val, 1 test)
        train_count = max(1, len(unique_dates) - 2)
        train_days = unique_dates[:train_count]
        val_days = [unique_dates[train_count]]
        test_days = unique_dates[train_count + 1 :]

    train_states = df[df["_date_str"].isin(train_days)].drop(columns=["_date_str"]).reset_index(drop=True)
    val_states = df[df["_date_str"].isin(val_days)].drop(columns=["_date_str"]).reset_index(drop=True)
    test_states = df[df["_date_str"].isin(test_days)].drop(columns=["_date_str"]).reset_index(drop=True)

    logger.info(
        f"Day-based split: Train days={train_days} ({len(train_states)} states), "
        f"Val days={val_days} ({len(val_states)} states), "
        f"Test days={test_days} ({len(test_states)} states)."
    )
    return train_states, val_states, test_states


def build_partitioned_temporal_dataset(
    train_states: pd.DataFrame,
    val_states: pd.DataFrame,
    test_states: pd.DataFrame,
    sequence_length: int = 5,
    forecast_horizon: int = 1,
    scale_features: bool = True,
    feature_columns: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Generate independent sequences and targets within each temporal partition.

    CRITICAL LEAKAGE PREVENTION:
    Sequences are generated independently per partition, ensuring that no sequence
    observation window in Train reaches into Val, and no target window crosses boundaries.
    Any feature scaler is fitted STRICTLY on X_train.

    Parameters
    ----------
    train_states, val_states, test_states : pd.DataFrame
        Partitioned state DataFrames.
    sequence_length : int
        Sequence length L (default: 5).
    forecast_horizon : int
        Forecast horizon H (default: 1).
    scale_features : bool
        Whether to standardize predictive features with StandardScaler fitted on Train.
    feature_columns : Optional[List[str]]
        Subset of predictive features to include.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing X arrays, target DataFrames, metadata, and fitted scaler.
    """
    builder = SequenceBuilder(
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        feature_columns=feature_columns,
    )

    # 1. Independent sequence generation per partition
    X_train, meta_train, tr_target_idx = builder.build_sequences(train_states)
    y_train = extract_future_targets(train_states, tr_target_idx)
    validate_target_alignment(X_train, y_train, meta_train)

    X_val, meta_val, val_target_idx = builder.build_sequences(val_states)
    y_val = extract_future_targets(val_states, val_target_idx)
    validate_target_alignment(X_val, y_val, meta_val)

    X_test, meta_test, te_target_idx = builder.build_sequences(test_states)
    y_test = extract_future_targets(test_states, te_target_idx)
    validate_target_alignment(X_test, y_test, meta_test)

    # 2. Strict Preprocessing Isolation: fit scaler ONLY on training data
    scaler: Optional[StandardScaler] = None
    if scale_features and len(X_train) > 0:
        scaler = StandardScaler()
        N_tr, L, D = X_train.shape
        # Flatten time dimension for scaler fitting: (N * L, D)
        X_train_flat = X_train.reshape(-1, D)
        scaler.fit(X_train_flat)

        X_train = scaler.transform(X_train_flat).reshape(N_tr, L, D).astype(np.float32)

        if len(X_val) > 0:
            N_val = len(X_val)
            X_val_flat = X_val.reshape(-1, D)
            X_val = scaler.transform(X_val_flat).reshape(N_val, L, D).astype(np.float32)

        if len(X_test) > 0:
            N_te = len(X_test)
            X_test_flat = X_test.reshape(-1, D)
            X_test = scaler.transform(X_test_flat).reshape(N_te, L, D).astype(np.float32)

        logger.info(f"Fitted StandardScaler strictly on {len(X_train_flat)} training state instances.")

    features = feature_columns or [c for c in STATE_PREDICTIVE_FEATURES if c in train_states.columns]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "meta_train": meta_train,
        "X_val": X_val,
        "y_val": y_val,
        "meta_val": meta_val,
        "X_test": X_test,
        "y_test": y_test,
        "meta_test": meta_test,
        "scaler": scaler,
        "features": features,
        "sequence_length": sequence_length,
        "forecast_horizon": forecast_horizon,
        "summary": {
            "train_sequences": len(X_train),
            "val_sequences": len(X_val),
            "test_sequences": len(X_test),
            "feature_dim": len(features),
            "scaler_fitted_on_train_only": scaler is not None,
        },
    }
