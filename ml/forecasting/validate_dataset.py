"""Temporal dataset validation and quality auditing for NETRA Stage 3A.

Verifies chronological monotonicity, absence of target/identifier leakage,
temporal boundary integrity, missing value hygiene, and class/stage distributions.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ml.forecasting.dataset_loader import RealDatasetLoader
from ml.forecasting.schema import (
    STATE_METADATA_COLUMNS,
    STATE_PREDICTIVE_FEATURES,
    STATE_TARGET_COLUMNS,
    TAXONOMY_DISCLAIMER,
)
from ml.forecasting.temporal_split import (
    build_partitioned_temporal_dataset,
    partition_states_chronologically,
)
from ml.forecasting.temporal_state import build_temporal_states

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("NETRA-DatasetValidator")


def audit_raw_flow_data(df: pd.DataFrame) -> Dict[str, Any]:
    """Perform quality checks on raw flow telemetry DataFrame."""
    issues: List[str] = []

    # 1. Timestamp validation
    has_time = "timestamp" in df.columns
    is_monotonic = False
    if has_time:
        t_series = pd.to_datetime(df["timestamp"], errors="coerce")
        null_times = int(t_series.isna().sum())
        if null_times > 0:
            issues.append(f"Contains {null_times} unparseable timestamp values.")
        clean_times = t_series.dropna()
        if not clean_times.empty:
            is_monotonic = bool(clean_times.is_monotonic_increasing)
            if not is_monotonic:
                issues.append("Timestamps are not monotonically increasing.")
    else:
        issues.append("Missing required 'timestamp' column.")

    # 2. Label validation
    has_label = "label" in df.columns
    label_dist = {}
    if has_label:
        label_dist = df["label"].value_counts().to_dict()
        null_labels = int(df["label"].isna().sum())
        if null_labels > 0:
            issues.append(f"Contains {null_labels} missing attack labels.")
    else:
        issues.append("Missing required 'label' column.")

    # 3. Numeric values, Infs, and NaNs
    num_cols = [c for c in df.columns if c not in {"timestamp", "label", "src_ip", "dst_ip"}]
    nan_count = int(df[num_cols].isna().sum().sum())
    inf_count = 0
    for c in num_cols:
        arr = pd.to_numeric(df[c], errors="coerce").to_numpy()
        inf_count += int(np.isinf(arr).sum())

    if nan_count > 0:
        issues.append(f"Detected {nan_count} NaN values in numeric columns.")
    if inf_count > 0:
        issues.append(f"Detected {inf_count} Infinite values in numeric columns.")

    # 4. Impossible values
    negative_counts = 0
    for c in ["tot_fwd_pkts", "tot_bwd_pkts", "tot_fwd_bytes", "tot_bwd_bytes", "flow_duration"]:
        if c in df.columns:
            neg = int((pd.to_numeric(df[c], errors="coerce") < 0).sum())
            if neg > 0:
                negative_counts += neg
                issues.append(f"Column '{c}' contains {neg} negative values.")

    return {
        "status": "PASS" if not issues else "FLAGGED",
        "total_rows": len(df),
        "is_chronologically_sorted": is_monotonic,
        "nan_count": nan_count,
        "inf_count": inf_count,
        "negative_anomaly_count": negative_counts,
        "label_distribution": label_dist,
        "issues_detected": issues,
    }


def audit_temporal_states(states_df: pd.DataFrame) -> Dict[str, Any]:
    """Audit temporal network state S_t DataFrame."""
    issues: List[str] = []

    # Check required feature columns
    missing_pred = [f for f in STATE_PREDICTIVE_FEATURES if f not in states_df.columns]
    if missing_pred:
        issues.append(f"Missing predictive state features: {missing_pred}")

    missing_meta = [f for f in STATE_METADATA_COLUMNS if f not in states_df.columns]
    if missing_meta:
        issues.append(f"Missing state metadata columns: {missing_meta}")

    missing_targets = [f for f in STATE_TARGET_COLUMNS if f not in states_df.columns]
    if missing_targets:
        issues.append(f"Missing state target columns: {missing_targets}")

    # Check window timestamps monotonicity
    is_monotonic = False
    if "window_start" in states_df.columns:
        w_start = pd.to_datetime(states_df["window_start"])
        is_monotonic = bool(w_start.is_monotonic_increasing)
        if not is_monotonic:
            issues.append("window_start is not strictly monotonically increasing.")

    # Check for empty / sparse windows
    empty_windows = int((states_df["flow_count"] == 0).sum()) if "flow_count" in states_df.columns else 0
    sparse_windows = int((states_df["flow_count"] < 3).sum()) if "flow_count" in states_df.columns else 0

    attack_windows = int(states_df["has_attack"].sum()) if "has_attack" in states_df.columns else 0
    total_windows = len(states_df)
    attack_ratio = float(attack_windows / total_windows) if total_windows > 0 else 0.0

    stage_dist = states_df["attack_stage"].value_counts().to_dict() if "attack_stage" in states_df.columns else {}

    return {
        "status": "PASS" if not issues else "FLAGGED",
        "total_windows": total_windows,
        "empty_windows": empty_windows,
        "sparse_windows": sparse_windows,
        "is_chronologically_sorted": is_monotonic,
        "attack_window_ratio": round(attack_ratio, 4),
        "attack_stage_distribution": stage_dist,
        "taxonomy_disclaimer": TAXONOMY_DISCLAIMER,
        "issues_detected": issues,
    }


def validate_forecasting_pipeline(
    raw_path: Union[str, Path],
    dataset_format: str = "CIC-IDS2017",
    window_size_sec: int = 60,
    stride_sec: int = 30,
    sequence_length: int = 5,
    forecast_horizon: int = 1,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute end-to-end Stage 3A validation and report generation."""
    logger.info(f"Initiating Stage 3A dataset validation for: {raw_path}")
    loader = RealDatasetLoader(dataset_format=dataset_format)
    flows_df, loader_summary = loader.load_dataset(raw_path)

    # 1. Audit Raw Telemetry
    raw_audit = audit_raw_flow_data(flows_df)

    # 2. Build Temporal States S_t
    states_df = build_temporal_states(
        flows_df,
        window_size_sec=window_size_sec,
        stride_sec=stride_sec,
    )
    states_audit = audit_temporal_states(states_df)

    # 3. Partition and Build Sequences
    train_s, val_s, test_s = partition_states_chronologically(states_df)
    dataset_bundle = build_partitioned_temporal_dataset(
        train_states=train_s,
        val_states=val_s,
        test_states=test_s,
        sequence_length=sequence_length,
        forecast_horizon=forecast_horizon,
        scale_features=True,
    )

    full_report: Dict[str, Any] = {
        "dataset_loader_summary": loader_summary,
        "raw_data_audit": raw_audit,
        "temporal_states_audit": states_audit,
        "partition_summary": dataset_bundle["summary"],
        "sequence_shapes": {
            "X_train": list(dataset_bundle["X_train"].shape),
            "X_val": list(dataset_bundle["X_val"].shape),
            "X_test": list(dataset_bundle["X_test"].shape),
        },
        "target_distributions": {
            "train_attack_binary": dataset_bundle["y_train"]["future_attack_binary"].value_counts().to_dict()
            if len(dataset_bundle["y_train"]) > 0
            else {},
            "test_attack_binary": dataset_bundle["y_test"]["future_attack_binary"].value_counts().to_dict()
            if len(dataset_bundle["y_test"]) > 0
            else {},
        },
    }

    if output_dir:
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        # Export states and metadata
        states_df.to_csv(out_path / "temporal_states.csv", index=False)
        if len(dataset_bundle["meta_train"]) > 0:
            dataset_bundle["meta_train"].to_csv(out_path / "train_sequence_metadata.csv", index=False)
            dataset_bundle["y_train"].to_csv(out_path / "train_forecast_targets.csv", index=False)
        with open(out_path / "stage3a_validation_report.json", "w") as f:
            json.dump(full_report, f, indent=2)
        logger.info(f"Stage 3A artifacts exported to: {out_path}")

    return full_report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate Stage 3A Temporal Forecasting Dataset")
    parser.add_argument("--input", default="data/samples/synthetic_network_flows.csv", help="Input dataset path")
    parser.add_argument("--format", default="CIC-IDS2017", help="Dataset format (CIC-IDS2017 / UNSW-NB15)")
    parser.add_argument("--output_dir", default="data/temporal", help="Directory to save artifacts")
    args = parser.parse_args()

    report = validate_forecasting_pipeline(
        raw_path=args.input,
        dataset_format=args.format,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2))
