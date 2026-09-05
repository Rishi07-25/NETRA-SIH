"""Comprehensive unit and pipeline tests for NETRA data preprocessing."""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from ml.preprocessing.clean_data import (
    clean_network_dataframe,
    identify_label_column,
    identify_timestamp_column,
    load_and_clean_dataset,
    normalize_column_names,
    normalize_labels,
)
from ml.preprocessing.feature_engineering import (
    compute_derived_flow_features,
    prepare_feature_matrix,
)
from ml.preprocessing.create_time_windows import create_sliding_time_windows

SAMPLE_CSV_PATH = Path("data/samples/synthetic_network_flows.csv")


def test_sample_csv_exists():
    """Verify that the synthetic sample dataset is present."""
    assert SAMPLE_CSV_PATH.exists(), f"Sample dataset not found at {SAMPLE_CSV_PATH}"


def test_normalize_column_names():
    """Test standardizing noisy column headers."""
    raw_cols = [" Destination Port", "Flow Duration ", "Total Fwd Packets", "Bwd Packet/s"]
    expected = ["destination_port", "flow_duration", "total_fwd_packets", "bwd_packet_s"]
    assert normalize_column_names(raw_cols) == expected


def test_label_normalization():
    """Test mapping polymorphic labels into canonical classes."""
    raw_labels = pd.Series(["BENIGN", "portscan", "SSH-Bruteforce", "ddos-synflood", "normal", "unknown_threat"])
    normalized = normalize_labels(raw_labels)
    assert normalized.iloc[0] == "BENIGN"
    assert normalized.iloc[1] == "Reconnaissance"
    assert normalized.iloc[2] == "Brute Force"
    assert normalized.iloc[3] == "DDoS"
    assert normalized.iloc[4] == "BENIGN"
    assert normalized.iloc[5] == "unknown_threat"


def test_clean_network_dataframe_duplicate_and_inf_handling():
    """Test handling duplicates, infinite values, and NaNs."""
    df = pd.DataFrame({
        "timestamp": ["2026-09-05 10:00:00", "2026-09-05 10:00:00", "2026-09-05 10:01:00"],
        "duration": [100.0, 100.0, np.inf],
        "packet_count": [10.0, 10.0, np.nan],
        "label": ["BENIGN", "BENIGN", "portscan"],
    })
    cleaned, label_col = clean_network_dataframe(df, drop_duplicates=True)
    assert len(cleaned) == 2  # duplicate removed
    assert label_col == "label"
    assert not np.isinf(cleaned["duration"]).any()
    assert not cleaned["duration"].isna().any()
    assert not cleaned["packet_count"].isna().any()
    assert cleaned["label"].iloc[1] == "Reconnaissance"


def test_load_and_clean_synthetic_sample():
    """Test full load and clean pipeline on the synthetic sample dataset."""
    df, label_col = load_and_clean_dataset(SAMPLE_CSV_PATH)
    assert not df.empty
    assert label_col == "label"
    assert "timestamp" in df.columns
    # Verify canonical classes
    assert set(df["label"].unique()).issubset({"BENIGN", "Reconnaissance", "Brute Force", "DDoS"})


def test_prepare_feature_matrix_leakage_and_types():
    """Test feature preparation ensures no label/metadata leakage and all-numeric output."""
    df, label_col = load_and_clean_dataset(SAMPLE_CSV_PATH)
    X, y, feature_names = prepare_feature_matrix(df, label_col=label_col)

    # Assert label is not in X
    assert label_col not in X.columns
    assert "timestamp" not in X.columns
    assert "src_ip" not in X.columns
    assert "dst_ip" not in X.columns

    # Assert shape correspondence
    assert len(X) == len(y)
    assert len(feature_names) == X.shape[1]

    # Assert numeric validity: no NaNs, no Infs
    assert not X.isna().any().any()
    assert not np.isinf(X.to_numpy()).any()


def test_create_sliding_time_windows():
    """Test sliding temporal window creation and chronological feature aggregation."""
    df, label_col = load_and_clean_dataset(SAMPLE_CSV_PATH)
    windows_df, labels_df = create_sliding_time_windows(
        df,
        window_size_sec=60,
        stride_sec=30,
        numeric_agg="mean",
    )

    assert not windows_df.empty
    assert not labels_df.empty
    assert len(windows_df) == len(labels_df)

    # Check window sequence
    assert "window_id" in windows_df.columns
    assert "flow_count" in windows_df.columns
    assert "dominant_label" in labels_df.columns
    assert "attack_flow_ratio" in labels_df.columns

    # Verify chronological ordering
    window_starts = windows_df["window_start"].tolist()
    assert window_starts == sorted(window_starts)
