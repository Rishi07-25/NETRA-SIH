"""Temporal Network State S_t Generation for NETRA Stage 3A.

Aggregates chronological flow telemetry into fixed-duration sliding windows [t, t + Δ)
to construct the predictive state vector S_t representing global network behavior.
Strictly separates predictive telemetry from ground-truth target metadata.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ml.forecasting.schema import (
    STATE_METADATA_COLUMNS,
    STATE_PREDICTIVE_FEATURES,
    STATE_TARGET_COLUMNS,
    map_label_to_attack_stage,
)
from ml.preprocessing.create_time_windows import (
    calculate_protocol_ratios,
    calculate_shannon_entropy,
)

logger = logging.getLogger(__name__)


def build_temporal_states(
    df: pd.DataFrame,
    window_size_sec: int = 60,
    stride_sec: int = 30,
    include_empty_windows: bool = False,
) -> pd.DataFrame:
    """Aggregate flow telemetry into chronological sliding network states S_t.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned network flows containing 'timestamp', 'label', and flow attributes.
    window_size_sec : int
        Window duration Δ in seconds (default: 60s).
    stride_sec : int
        Window slide step in seconds (default: 30s).
    include_empty_windows : bool
        Whether to generate zero-filled state records for silent periods.

    Returns
    -------
    pd.DataFrame
        DataFrame containing window metadata, predictive state S_t features,
        and isolated target labels.
    """
    if "timestamp" not in df.columns:
        raise ValueError("DataFrame must contain 'timestamp' column.")
    if "label" not in df.columns:
        raise ValueError("DataFrame must contain 'label' column.")

    data = df.copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"])
    data = data.sort_values(by="timestamp").reset_index(drop=True)
    if data.empty:
        return pd.DataFrame(columns=STATE_METADATA_COLUMNS + STATE_PREDICTIVE_FEATURES + STATE_TARGET_COLUMNS)

    start_time = data["timestamp"].min()
    end_time = data["timestamp"].max()

    window_delta = pd.Timedelta(seconds=window_size_sec)
    stride_delta = pd.Timedelta(seconds=stride_sec)

    records: List[Dict[str, Any]] = []
    window_index = 0
    current_start = start_time

    while current_start <= end_time:
        current_end = current_start + window_delta

        # Half-open interval [t, t + Δ) prevents future leakage
        mask = (data["timestamp"] >= current_start) & (data["timestamp"] < current_end)
        slice_df = data.loc[mask]

        if slice_df.empty:
            if include_empty_windows:
                # Zero-filled baseline state
                empty_record: Dict[str, Any] = {feat: 0.0 for feat in STATE_PREDICTIVE_FEATURES}
                empty_record["window_id"] = window_index
                empty_record["window_start"] = current_start
                empty_record["window_end"] = current_end
                empty_record["dominant_label"] = "BENIGN"
                empty_record["has_attack"] = False
                empty_record["attack_flow_ratio"] = 0.0
                empty_record["attack_stage"] = "Normal"
                records.append(empty_record)
                window_index += 1
            current_start += stride_delta
            continue

        flow_count = len(slice_df)

        # 1. Continuous feature aggregations
        def _safe_mean(col: str) -> float:
            return float(slice_df[col].mean()) if col in slice_df.columns else 0.0

        # Flow volume metrics
        tot_fwd_pkts = _safe_mean("tot_fwd_pkts")
        tot_bwd_pkts = _safe_mean("tot_bwd_pkts")
        tot_fwd_bytes = _safe_mean("tot_fwd_bytes")
        tot_bwd_bytes = _safe_mean("tot_bwd_bytes")
        avg_pkt_size = _safe_mean("avg_pkt_size")

        # Rate dynamics
        flow_duration = _safe_mean("flow_duration")
        flow_pkt_rate = _safe_mean("flow_pkt_rate")
        flow_byte_rate = _safe_mean("flow_byte_rate")

        # Control flags (mean counts per flow)
        syn_flag_cnt = _safe_mean("syn_flag_cnt")
        rst_flag_cnt = _safe_mean("rst_flag_cnt")
        ack_flag_cnt = _safe_mean("ack_flag_cnt")

        # Ratio dynamics
        fwd_bwd_pkt_ratio = _safe_mean("fwd_bwd_pkt_ratio")
        fwd_bwd_byte_ratio = _safe_mean("fwd_bwd_byte_ratio")

        # Semantic endpoint & protocol distributions
        unique_src_ports = int(slice_df["src_port"].nunique()) if "src_port" in slice_df.columns else 0
        unique_dst_ports = int(slice_df["dst_port"].nunique()) if "dst_port" in slice_df.columns else 0
        dst_port_entropy = calculate_shannon_entropy(slice_df["dst_port"]) if "dst_port" in slice_df.columns else 0.0

        if "protocol" in slice_df.columns:
            tcp_ratio, udp_ratio = calculate_protocol_ratios(slice_df["protocol"])
        else:
            tcp_ratio, udp_ratio = 0.0, 0.0

        # 2. Window Ground-Truth Labels & Operational Stage
        label_counts = slice_df["label"].value_counts()
        dominant_label = str(label_counts.index[0]) if not label_counts.empty else "BENIGN"
        has_attack = bool(any(str(lbl).upper() != "BENIGN" for lbl in label_counts.index))
        attack_flow_ratio = float(1.0 - (label_counts.get("BENIGN", 0) / flow_count))
        attack_stage = map_label_to_attack_stage(dominant_label)

        # Build combined state record
        record: Dict[str, Any] = {
            # Metadata
            "window_id": window_index,
            "window_start": current_start,
            "window_end": current_end,
            # S_t Predictive Features
            "flow_count": float(flow_count),
            "tot_fwd_pkts": tot_fwd_pkts,
            "tot_bwd_pkts": tot_bwd_pkts,
            "tot_fwd_bytes": tot_fwd_bytes,
            "tot_bwd_bytes": tot_bwd_bytes,
            "avg_pkt_size": avg_pkt_size,
            "flow_duration": flow_duration,
            "flow_pkt_rate": flow_pkt_rate,
            "flow_byte_rate": flow_byte_rate,
            "syn_flag_cnt": syn_flag_cnt,
            "rst_flag_cnt": rst_flag_cnt,
            "ack_flag_cnt": ack_flag_cnt,
            "fwd_bwd_pkt_ratio": fwd_bwd_pkt_ratio,
            "fwd_bwd_byte_ratio": fwd_bwd_byte_ratio,
            "unique_src_ports": float(unique_src_ports),
            "unique_dst_ports": float(unique_dst_ports),
            "dst_port_entropy": float(dst_port_entropy),
            "tcp_ratio": float(tcp_ratio),
            "udp_ratio": float(udp_ratio),
            # Isolated Target Labels
            "dominant_label": dominant_label,
            "has_attack": has_attack,
            "attack_flow_ratio": attack_flow_ratio,
            "attack_stage": attack_stage,
        }

        records.append(record)
        window_index += 1
        current_start += stride_delta

    states_df = pd.DataFrame(records)
    logger.info(
        f"Built {len(states_df)} temporal network states (window: {window_size_sec}s, stride: {stride_sec}s)."
    )
    return states_df


def separate_predictive_and_target_features(
    states_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Strictly partition temporal state dataframe into predictive inputs, targets, and metadata.

    Guarantees that ground-truth attack labels and window IDs never contaminate
    predictive state vectors.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        predictive_df (S_t features), targets_df (labels/stage), metadata_df (timestamps/IDs)
    """
    predictive_cols = [c for c in STATE_PREDICTIVE_FEATURES if c in states_df.columns]
    target_cols = [c for c in STATE_TARGET_COLUMNS if c in states_df.columns]
    meta_cols = [c for c in STATE_METADATA_COLUMNS if c in states_df.columns]

    predictive_df = states_df[predictive_cols].copy()
    targets_df = states_df[target_cols].copy()
    metadata_df = states_df[meta_cols].copy()

    return predictive_df, targets_df, metadata_df
