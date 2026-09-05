"""Temporal sliding window generator for sequential network risk forecasting.

Groups chronological flow telemetry into sliding time windows and aggregates
behavioral indicators without leaking future temporal information.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ml.preprocessing.clean_data import identify_label_column, identify_timestamp_column

logger = logging.getLogger(__name__)


def create_sliding_time_windows(
    df: pd.DataFrame,
    timestamp_col: Optional[str] = None,
    label_col: Optional[str] = None,
    window_size_sec: int = 60,
    stride_sec: int = 30,
    numeric_agg: str = "mean",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Generate chronological sliding time windows from network flow records.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned network flow dataset containing timestamps.
    timestamp_col : Optional[str]
        Timestamp column name. Auto-detected if None.
    label_col : Optional[str]
        Label column name. Auto-detected if None.
    window_size_sec : int
        Duration of each temporal window in seconds (e.g., 60s).
    stride_sec : int
        Step size between consecutive windows in seconds (e.g., 30s).
    numeric_agg : str
        Aggregation function for numeric metrics ('mean', 'sum', 'median').

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        windows_df : Aggregated window feature matrix with window metadata.
        window_labels_df : Summarized attack label statistics per window.
    """
    data = df.copy()

    # Identify Timestamp
    time_col = timestamp_col if timestamp_col and timestamp_col in data.columns else identify_timestamp_column(data)
    if time_col is None or time_col not in data.columns:
        raise ValueError("Timestamp column not found; cannot create temporal windows.")

    # Identify Label
    target_col = label_col if label_col and label_col in data.columns else identify_label_column(data)
    if target_col is None or target_col not in data.columns:
        raise ValueError("Label column not found; cannot track threat labels across windows.")

    # Convert to datetime and sort strictly chronologically
    data[time_col] = pd.to_datetime(data[time_col])
    data = data.sort_values(by=time_col).reset_index(drop=True)

    start_time = data[time_col].min()
    end_time = data[time_col].max()

    window_delta = pd.Timedelta(seconds=window_size_sec)
    stride_delta = pd.Timedelta(seconds=stride_sec)

    # Separate numeric features from identifiers/labels
    non_feature_cols = {time_col, target_col, "src_ip", "dst_ip", "source_ip", "destination_ip"}
    numeric_feature_cols = [
        c for c in data.columns if c not in non_feature_cols and pd.api.types.is_numeric_dtype(data[c])
    ]

    window_records = []
    label_records = []

    current_window_start = start_time
    window_index = 0

    while current_window_start + window_delta <= end_time + stride_delta:
        current_window_end = current_window_start + window_delta

        # Filter strictly within the historical window [start, end)
        # Prevents future leakage into the current slice
        mask = (data[time_col] >= current_window_start) & (data[time_col] < current_window_end)
        window_slice = data.loc[mask]

        if not window_slice.empty:
            flow_count = len(window_slice)

            # Compute aggregated features
            agg_series = getattr(window_slice[numeric_feature_cols], numeric_agg)()
            window_dict = agg_series.to_dict()
            window_dict["window_id"] = window_index
            window_dict["window_start"] = current_window_start
            window_dict["window_end"] = current_window_end
            window_dict["flow_count"] = flow_count

            # Extract label statistics within this window
            label_counts = window_slice[target_col].value_counts()
            has_attack = any(lbl != "BENIGN" for lbl in label_counts.index)
            dominant_label = label_counts.index[0] if not label_counts.empty else "BENIGN"
            attack_ratio = 1.0 - (label_counts.get("BENIGN", 0) / flow_count)

            label_dict = {
                "window_id": window_index,
                "window_start": current_window_start,
                "window_end": current_window_end,
                "dominant_label": dominant_label,
                "has_attack": bool(has_attack),
                "attack_flow_ratio": float(attack_ratio),
                "total_flows": flow_count,
            }

            window_records.append(window_dict)
            label_records.append(label_dict)
            window_index += 1

        current_window_start += stride_delta

    if not window_records:
        logger.warning("No windows produced. Check window_size_sec vs dataset duration.")
        return pd.DataFrame(), pd.DataFrame()

    windows_df = pd.DataFrame(window_records)
    window_labels_df = pd.DataFrame(label_records)

    logger.info(
        f"Generated {len(windows_df)} time windows (size: {window_size_sec}s, stride: {stride_sec}s)."
    )
    return windows_df, window_labels_df
