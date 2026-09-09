"""Temporal sliding window generator for sequential network risk forecasting.

Groups chronological flow telemetry into sliding time windows and aggregates
behavioral indicators without leaking future temporal information.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd

from ml.preprocessing.clean_data import identify_label_column, identify_timestamp_column

logger = logging.getLogger(__name__)

# Port and protocol columns and aliases that must NEVER be averaged directly
PORT_AND_PROTOCOL_EXCLUSIONS: Set[str] = {
    "src_port",
    "source_port",
    "sport",
    "dst_port",
    "destination_port",
    "dport",
    "protocol",
    "proto",
}


def calculate_shannon_entropy(values: pd.Series) -> float:
    """Calculate Shannon entropy H(X) = -sum(p * log2(p)) over a categorical/discrete series.

    Returns 0.0 if empty or only one unique value exists.
    """
    clean_series = values.dropna()
    if clean_series.empty or len(clean_series) <= 1:
        return 0.0

    counts = clean_series.value_counts()
    if len(counts) <= 1:
        return 0.0

    probabilities = counts / len(clean_series)
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    if math.isnan(entropy) or math.isinf(entropy):
        return 0.0
    return max(0.0, entropy)


def calculate_protocol_ratios(proto_series: pd.Series) -> Tuple[float, float]:
    """Calculate the ratio of TCP and UDP flows in the given protocol series.

    Supports numeric representation (TCP=6, UDP=17) and string ('tcp', 'udp') representations.
    Returns (tcp_ratio, udp_ratio).
    """
    clean_proto = proto_series.dropna()
    if clean_proto.empty:
        return 0.0, 0.0

    total_valid = len(clean_proto)
    tcp_count = 0
    udp_count = 0

    for val in clean_proto:
        # Numeric check
        try:
            num_val = int(val)
            if num_val == 6:
                tcp_count += 1
            elif num_val == 17:
                udp_count += 1
            continue
        except (ValueError, TypeError):
            pass

        # String check
        s_val = str(val).strip().lower()
        if s_val == "tcp":
            tcp_count += 1
        elif s_val == "udp":
            udp_count += 1

    tcp_ratio = tcp_count / total_valid if total_valid > 0 else 0.0
    udp_ratio = udp_count / total_valid if total_valid > 0 else 0.0
    return float(tcp_ratio), float(udp_ratio)


def find_column_by_aliases(df: pd.DataFrame, aliases: Set[str]) -> Optional[str]:
    """Find the column name matching any alias in the provided alias set."""
    for col in df.columns:
        if col.lower() in aliases:
            return col
    return None


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
        windows_df : Aggregated window feature matrix containing continuous traffic metrics
                     and semantic temporal features (unique ports, entropy, protocol ratios).
        window_labels_df : Summarized attack label statistics per window (isolated ground truth).
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

    # Detect port and protocol columns to handle with dedicated semantic aggregations
    src_port_col = find_column_by_aliases(data, {"src_port", "source_port", "sport"})
    dst_port_col = find_column_by_aliases(data, {"dst_port", "destination_port", "dport"})
    proto_col = find_column_by_aliases(data, {"protocol", "proto"})

    # Build non_feature exclusion set (IDs, IPs, labels, timestamps, and ports/proto)
    non_feature_cols = {
        time_col,
        target_col,
        "src_ip",
        "dst_ip",
        "source_ip",
        "destination_ip",
    }
    non_feature_cols.update(PORT_AND_PROTOCOL_EXCLUSIONS)

    # Numeric features that are semantically appropriate for direct arithmetic aggregation
    numeric_feature_cols = [
        c for c in data.columns if c.lower() not in non_feature_cols and pd.api.types.is_numeric_dtype(data[c])
    ]

    window_records = []
    label_records = []

    current_window_start = start_time
    window_index = 0

    # Ensure windows cover from start_time up to and including the last flow event timestamp
    while current_window_start <= end_time:
        current_window_end = current_window_start + window_delta

        # Strict half-open interval [start, end)
        # Guarantees start is included and end is excluded, preventing future leakage
        mask = (data[time_col] >= current_window_start) & (data[time_col] < current_window_end)
        window_slice = data.loc[mask]

        if not window_slice.empty:
            flow_count = len(window_slice)

            # Compute standard numeric aggregations on legitimate continuous metrics
            if numeric_feature_cols:
                agg_series = getattr(window_slice[numeric_feature_cols], numeric_agg)()
                window_dict = agg_series.to_dict()
            else:
                window_dict = {}

            # Metadata identifiers for the window
            window_dict["window_id"] = window_index
            window_dict["window_start"] = current_window_start
            window_dict["window_end"] = current_window_end
            window_dict["flow_count"] = flow_count

            # Semantic Port Metrics (Distinct Counts & Entropy)
            if src_port_col and src_port_col in window_slice.columns:
                window_dict["unique_src_ports"] = int(window_slice[src_port_col].dropna().nunique())
            else:
                window_dict["unique_src_ports"] = 0

            if dst_port_col and dst_port_col in window_slice.columns:
                window_dict["unique_dst_ports"] = int(window_slice[dst_port_col].dropna().nunique())
                window_dict["dst_port_entropy"] = calculate_shannon_entropy(window_slice[dst_port_col])
            else:
                window_dict["unique_dst_ports"] = 0
                window_dict["dst_port_entropy"] = 0.0

            # Semantic Protocol Ratios
            if proto_col and proto_col in window_slice.columns:
                tcp_ratio, udp_ratio = calculate_protocol_ratios(window_slice[proto_col])
                window_dict["tcp_ratio"] = tcp_ratio
                window_dict["udp_ratio"] = udp_ratio
            else:
                window_dict["tcp_ratio"] = 0.0
                window_dict["udp_ratio"] = 0.0

            # Extract isolated label statistics for this window
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
