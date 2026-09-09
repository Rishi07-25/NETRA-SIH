"""Feature engineering and feature matrix preparation for network-flow modeling.

Ensures strict separation of features and labels to prevent target leakage.
Supports tabular feature matrices for standard classification/anomaly models.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from ml.preprocessing.clean_data import identify_label_column, identify_timestamp_column

logger = logging.getLogger(__name__)

# Metadata and identifier columns that should typically be excluded from ML feature matrix
DEFAULT_METADATA_COLUMNS: Set[str] = {
    "src_ip",
    "dst_ip",
    "source_ip",
    "destination_ip",
    "source_mac",
    "destination_mac",
    "flow_id",
    "id",
    "session_id",
    "timestamp",
    "time",
    "start_time",
    "stime",
    "ltime",
    "datetime",
    "src_port",
    "source_port",
    "sport",
}


def compute_derived_flow_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute standard derived network flow features if prerequisite fields exist.

    Calculates:
    - packet_size_ratio: forward to backward packet ratio
    - byte_ratio: forward to backward byte ratio
    - avg_packet_size: total bytes divided by total packets
    """
    enhanced = df.copy()

    # Determine forward/backward packet fields (CIC-IDS style)
    fwd_pkts = None
    bwd_pkts = None
    fwd_bytes = None
    bwd_bytes = None
    duration = None

    for col in enhanced.columns:
        if "fwd_pkt" in col or "tot_fwd_pkts" in col:
            fwd_pkts = col
        elif "bwd_pkt" in col or "tot_bwd_pkts" in col:
            bwd_pkts = col
        elif "fwd_byte" in col or "tot_fwd_bytes" in col:
            fwd_bytes = col
        elif "bwd_byte" in col or "tot_bwd_bytes" in col:
            bwd_bytes = col
        elif "flow_duration" in col or "duration" in col:
            duration = col

    # Derived Packet Ratio
    if fwd_pkts and bwd_pkts:
        denominator = enhanced[bwd_pkts].replace(0, 1)
        enhanced["fwd_bwd_pkt_ratio"] = enhanced[fwd_pkts] / denominator

    # Derived Byte Ratio
    if fwd_bytes and bwd_bytes:
        denominator = enhanced[bwd_bytes].replace(0, 1)
        enhanced["fwd_bwd_byte_ratio"] = enhanced[fwd_bytes] / denominator

    # Average Packet Size
    if fwd_pkts and bwd_pkts and fwd_bytes and bwd_bytes:
        tot_pkts = (enhanced[fwd_pkts] + enhanced[bwd_pkts]).replace(0, 1)
        tot_bytes = enhanced[fwd_bytes] + enhanced[bwd_bytes]
        enhanced["avg_pkt_size"] = tot_bytes / tot_pkts

    # Handle any generated NaNs/Infs
    new_cols = [c for c in ["fwd_bwd_pkt_ratio", "fwd_bwd_byte_ratio", "avg_pkt_size"] if c in enhanced.columns]
    for c in new_cols:
        enhanced[c] = enhanced[c].replace([np.inf, -np.inf], 0.0).fillna(0.0)

    return enhanced


def prepare_feature_matrix(
    df: pd.DataFrame,
    label_col: Optional[str] = None,
    exclude_columns: Optional[List[str]] = None,
    compute_derived: bool = True,
    categorical_strategy: str = "one_hot",
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Extract and validate numeric feature matrix X and target label y.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned network flow dataframe.
    label_col : Optional[str]
        Target label column name. Auto-detected if None.
    exclude_columns : Optional[List[str]]
        Explicit list of metadata/identifier columns to exclude from X.
    compute_derived : bool
        Whether to calculate ratios and flow summary metrics.
    categorical_strategy : str
        Method for encoding non-numeric features ('one_hot' or 'drop').

    Returns
    -------
    Tuple[pd.DataFrame, pd.Series, List[str]]
        X : Feature matrix (numeric pd.DataFrame)
        y : Target labels (pd.Series)
        feature_names : List of feature column names in X
    """
    data = df.copy()

    # Identify Label Column
    target_col = label_col if label_col and label_col in data.columns else identify_label_column(data)
    if target_col is None or target_col not in data.columns:
        raise ValueError(f"Target label column '{target_col}' not found in dataframe.")

    y = data[target_col].copy()

    # Derive additional statistical features if requested
    if compute_derived:
        data = compute_derived_flow_features(data)

    # Establish Exclusions to Prevent Data Leakage
    exclusions = set(DEFAULT_METADATA_COLUMNS)
    exclusions.add(target_col)

    # Also exclude detected timestamp column
    time_col = identify_timestamp_column(data)
    if time_col:
        exclusions.add(time_col)

    if exclude_columns:
        exclusions.update(c.lower() for c in exclude_columns)

    feature_cols = [col for col in data.columns if col.lower() not in exclusions]

    # Partition features
    features_df = data[feature_cols].copy()

    # Handle Categorical Columns
    numeric_cols = []
    categorical_cols = []

    for col in features_df.columns:
        if pd.api.types.is_numeric_dtype(features_df[col]):
            numeric_cols.append(col)
        else:
            categorical_cols.append(col)

    if categorical_cols:
        logger.info(f"Detected categorical columns: {categorical_cols} (strategy: {categorical_strategy})")
        if categorical_strategy == "one_hot":
            # One-hot encode with drop_first=False
            encoded = pd.get_dummies(features_df[categorical_cols], drop_first=False, dtype=float)
            X = pd.concat([features_df[numeric_cols], encoded], axis=1)
        elif categorical_strategy == "drop":
            X = features_df[numeric_cols].copy()
        else:
            raise ValueError(f"Unknown categorical strategy: {categorical_strategy}")
    else:
        X = features_df[numeric_cols].copy()

    # Final Validation: ensure all numeric, no NaNs or Infs
    X = X.replace([np.inf, -np.inf], np.nan)
    if X.isna().any().any():
        logger.warning("Remaining NaNs detected in feature matrix X; filling with 0.0")
        X = X.fillna(0.0)

    feature_names = list(X.columns)
    logger.info(f"Feature matrix prepared with shape {X.shape} ({len(feature_names)} features).")
    return X, y, feature_names
