"""Data validation and train/test splitting utilities for NETRA Stage 2 Threat Engine.

Validates input feature and label datasets, ensures alignment, checks for leakage,
and produces deterministic, stratified splits with safety fallbacks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

# Disallowed identifier or leak-prone column names in feature matrix X
PROHIBITED_FEATURE_COLUMNS: Set[str] = {
    "label",
    "target",
    "attack_cat",
    "class",
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "timestamp",
    "time",
    "has_attack",
    "attack_flow_ratio",
    "dominant_label",
}


def validate_feature_data(
    X: pd.DataFrame,
    y: Optional[pd.Series | pd.DataFrame] = None,
    expected_features: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Validate feature matrix X and target label y for machine learning consumption.

    Parameters
    ----------
    X : pd.DataFrame
        Input feature matrix.
    y : Optional[pd.Series or pd.DataFrame]
        Target attack labels.
    expected_features : Optional[List[str]]
        Specific list and ordering of feature columns expected by a fitted model.

    Returns
    -------
    Tuple[pd.DataFrame, Optional[pd.Series]]
        Validated feature matrix and squeezed target label series.

    Raises
    ------
    ValueError
        If validation constraints are violated.
    """
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)

    if X.empty:
        raise ValueError("Feature matrix X cannot be empty.")

    # Check for target or identifier leakage in X
    leaked = [col for col in X.columns if col.lower() in PROHIBITED_FEATURE_COLUMNS]
    if leaked:
        raise ValueError(
            f"Prohibited target/identifier columns detected in feature matrix X: {leaked}. "
            "Exclusion is required to prevent data leakage."
        )

    # Validate against expected feature schema if provided
    if expected_features is not None:
        missing = [f for f in expected_features if f not in X.columns]
        if missing:
            raise ValueError(f"Feature matrix is missing required features: {missing}")
        # Enforce exact feature ordering
        X = X[expected_features].copy()

    # Numeric & finite checks
    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            raise ValueError(f"Feature column '{col}' contains non-numeric data type: {X[col].dtype}")

    # Check for NaNs and Infinities
    if X.isna().any().any():
        raise ValueError("Feature matrix X contains NaN values. Preprocessing/imputation required.")

    if np.isinf(X.to_numpy()).any():
        raise ValueError("Feature matrix X contains Infinite values. Preprocessing required.")

    # Validate label series y if provided
    y_series = None
    if y is not None:
        if isinstance(y, pd.DataFrame):
            if y.shape[1] != 1:
                raise ValueError(f"Target y DataFrame must have exactly 1 column, got shape {y.shape}")
            y_series = y.iloc[:, 0]
        else:
            y_series = pd.Series(y)

        if len(X) != len(y_series):
            raise ValueError(
                f"Row count mismatch between features X ({len(X)}) and target y ({len(y_series)})."
            )

        if y_series.isna().any():
            raise ValueError("Target labels y contains missing/NaN values.")

    return X, y_series


def split_threat_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.20,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a deterministic train/test split with safe stratification.

    Parameters
    ----------
    X : pd.DataFrame
        Validated feature matrix.
    y : pd.Series
        Target labels.
    test_size : float
        Proportion of dataset to include in the test split.
    random_state : int
        Deterministic seed for reproducibility.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]
        X_train, X_test, y_train, y_test
    """
    X_val, y_val = validate_feature_data(X, y)

    class_counts = y_val.value_counts()
    min_class_count = class_counts.min()

    # Stratified split requires at least 2 samples per class
    can_stratify = min_class_count >= 2
    if not can_stratify:
        logger.warning(
            f"Smallest class has only {min_class_count} sample(s). "
            "Stratification is not mathematically possible; falling back to non-stratified random split."
        )

    stratify_target = y_val if can_stratify else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_val,
        y_val,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_target,
    )

    logger.info(
        f"Data split complete: Train shape: {X_train.shape}, Test shape: {X_test.shape} "
        f"(stratified={can_stratify}, random_state={random_state})"
    )
    return X_train, X_test, y_train, y_test


def load_and_validate_features(
    features_path: Union[str, Path] = "data/features/features_X.csv",
    labels_path: Union[str, Path] = "data/features/labels_y.csv",
) -> Tuple[pd.DataFrame, pd.Series]:
    """Load and validate features_X and labels_y from CSV paths."""
    f_path = Path(features_path)
    l_path = Path(labels_path)

    if not f_path.exists():
        raise FileNotFoundError(f"Features file not found at: {f_path}")
    if not l_path.exists():
        raise FileNotFoundError(f"Labels file not found at: {l_path}")

    X = pd.read_csv(f_path)
    y = pd.read_csv(l_path)

    return validate_feature_data(X, y)
