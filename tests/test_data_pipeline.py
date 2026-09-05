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
from ml.preprocessing.create_time_windows import (
    calculate_protocol_ratios,
    calculate_shannon_entropy,
    create_sliding_time_windows,
)

SAMPLE_CSV_PATH = Path("data/samples/synthetic_network_flows.csv")


# =============================================================================
# EXISTING 7 TESTS (PRESERVED & EXPANDED)
# =============================================================================

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
    assert set(df["label"].unique()).issubset({"BENIGN", "Reconnaissance", "Brute Force", "DDoS"})


def test_prepare_feature_matrix_leakage_and_types():
    """Test feature preparation ensures no label/metadata leakage and all-numeric output."""
    df, label_col = load_and_clean_dataset(SAMPLE_CSV_PATH)
    X, y, feature_names = prepare_feature_matrix(df, label_col=label_col)

    # Assert label and metadata are not in X
    assert label_col not in X.columns
    assert "timestamp" not in X.columns
    assert "src_ip" not in X.columns
    assert "dst_ip" not in X.columns
    assert "src_port" not in X.columns

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


# =============================================================================
# NEW EDGE CASE & AUDIT FIX TESTS
# =============================================================================

def test_temporal_port_aggregation_no_averages():
    """Test Fix #1: Ports are aggregated into unique counts, NOT arithmetic means."""
    df = pd.DataFrame({
        "timestamp": ["2026-09-05 10:00:10", "2026-09-05 10:00:20", "2026-09-05 10:00:30"],
        "src_port": [50000, 50001, 50002],
        "dst_port": [22, 80, 443],
        "flow_duration": [100.0, 200.0, 300.0],
        "label": ["BENIGN", "BENIGN", "BENIGN"],
    })
    windows_df, labels_df = create_sliding_time_windows(df, window_size_sec=60, stride_sec=30)

    assert len(windows_df) == 1
    # Assert unique port counts exist
    assert windows_df["unique_src_ports"].iloc[0] == 3
    assert windows_df["unique_dst_ports"].iloc[0] == 3
    # Assert raw arithmetic averages of ports are EXCLUDED
    assert "src_port" not in windows_df.columns
    assert "dst_port" not in windows_df.columns


def test_destination_port_entropy():
    """Test Shannon entropy calculation on destination port distributions."""
    # Uniform 2-value distribution: p = [0.5, 0.5] -> H = 1.0 bit
    ports_uniform = pd.Series([80, 80, 443, 443])
    entropy_uniform = calculate_shannon_entropy(ports_uniform)
    assert pytest.approx(entropy_uniform, abs=1e-3) == 1.0

    # Single-port distribution -> H = 0.0
    ports_single = pd.Series([443, 443, 443])
    assert calculate_shannon_entropy(ports_single) == 0.0

    # Empty series -> H = 0.0
    assert calculate_shannon_entropy(pd.Series([], dtype=float)) == 0.0


def test_protocol_ratios_numeric_and_string():
    """Test TCP/UDP ratio calculations with numeric and string inputs."""
    # Numeric protocol: 6=TCP, 17=UDP
    num_proto = pd.Series([6, 6, 17, 17])
    tcp_r, udp_r = calculate_protocol_ratios(num_proto)
    assert pytest.approx(tcp_r) == 0.5
    assert pytest.approx(udp_r) == 0.5

    # Case-insensitive strings: "tcp", "TCP", "udp", "UDP"
    str_proto = pd.Series(["tcp", "TCP", "udp", "UDP"])
    tcp_s, udp_s = calculate_protocol_ratios(str_proto)
    assert pytest.approx(tcp_s) == 0.5
    assert pytest.approx(udp_s) == 0.5

    # Other/ICMP handling
    other_proto = pd.Series([1, "icmp"])
    tcp_o, udp_o = calculate_protocol_ratios(other_proto)
    assert tcp_o == 0.0
    assert udp_o == 0.0


def test_no_protocol_average_in_windows():
    """Verify that protocol is NEVER aggregated as an arithmetic mean."""
    df = pd.DataFrame({
        "timestamp": ["2026-09-05 10:00:10", "2026-09-05 10:00:20"],
        "protocol": [6, 17],
        "flow_duration": [100.0, 200.0],
        "label": ["BENIGN", "BENIGN"],
    })
    windows_df, _ = create_sliding_time_windows(df, window_size_sec=60, stride_sec=30)
    assert "protocol" not in windows_df.columns
    assert "tcp_ratio" in windows_df.columns
    assert "udp_ratio" in windows_df.columns
    assert pytest.approx(windows_df["tcp_ratio"].iloc[0]) == 0.5


def test_unix_epoch_timestamp_handling():
    """Test Fix #2: Correctly parse Unix epoch seconds without 1970 collapse."""
    # 1772719200 corresponds to 2026-03-05
    epoch_seconds = [1772719200.0, 1772719210.0, 1772719220.0]
    df = pd.DataFrame({
        "stime": epoch_seconds,
        "flow_duration": [10.0, 20.0, 30.0],
        "label": ["BENIGN", "BENIGN", "BENIGN"],
    })
    cleaned, _ = clean_network_dataframe(df)
    assert pd.api.types.is_datetime64_any_dtype(cleaned["stime"])
    assert cleaned["stime"].iloc[0].year >= 2026


def test_zero_duration_flows_remain_finite():
    """Verify flow_duration = 0 does not produce infinite or NaN derived metrics."""
    df = pd.DataFrame({
        "timestamp": ["2026-09-05 10:00:00", "2026-09-05 10:00:10"],
        "flow_duration": [0.0, 0.0],
        "tot_fwd_pkts": [0, 0],
        "tot_bwd_pkts": [0, 0],
        "tot_fwd_bytes": [0, 0],
        "tot_bwd_bytes": [0, 0],
        "label": ["BENIGN", "BENIGN"],
    })
    cleaned, label_col = clean_network_dataframe(df)
    X, y, _ = prepare_feature_matrix(cleaned, label_col=label_col)

    assert not X.isna().any().any()
    assert not np.isinf(X.to_numpy()).any()


def test_window_half_open_boundary():
    """Verify strict [start, end) semantics: start included, end excluded."""
    df = pd.DataFrame({
        "timestamp": [
            "2026-09-05 10:00:00",  # Exact start of window 0 -> INCLUDED in window 0
            "2026-09-05 10:00:59",  # Inside window 0 -> INCLUDED in window 0
            "2026-09-05 10:01:00",  # Exact end of window 0 -> EXCLUDED from window 0
        ],
        "flow_duration": [100.0, 200.0, 300.0],
        "label": ["BENIGN", "BENIGN", "BENIGN"],
    })
    windows_df, _ = create_sliding_time_windows(df, window_size_sec=60, stride_sec=60)
    # Window 0: [10:00:00, 10:01:00) -> 2 flows
    assert windows_df["flow_count"].iloc[0] == 2
    # Window 1: [10:01:00, 10:02:00) -> 1 flow
    assert windows_df["flow_count"].iloc[1] == 1


def test_empty_silent_windows_handled_gracefully():
    """Verify traffic gaps larger than window size do not crash the pipeline."""
    df = pd.DataFrame({
        "timestamp": [
            "2026-09-05 10:00:00",
            "2026-09-05 10:05:00",  # 5-minute gap
        ],
        "flow_duration": [100.0, 100.0],
        "label": ["BENIGN", "BENIGN"],
    })
    # Window size 60s, stride 60s
    windows_df, labels_df = create_sliding_time_windows(df, window_size_sec=60, stride_sec=60)
    assert not windows_df.empty
    assert len(windows_df) == 2  # Generates entries for populated intervals without crashing


def test_strict_label_isolation_in_features():
    """Verify target_col, has_attack, and attack_flow_ratio NEVER enter feature outputs."""
    df, label_col = load_and_clean_dataset(SAMPLE_CSV_PATH)
    X, _, _ = prepare_feature_matrix(df, label_col=label_col)
    windows_df, labels_df = create_sliding_time_windows(df, window_size_sec=60, stride_sec=30)

    forbidden = {"label", "has_attack", "attack_flow_ratio", "dominant_label"}
    for col in forbidden:
        assert col not in X.columns, f"Leaked {col} in X"
        assert col not in windows_df.columns, f"Leaked {col} in windows_df"

    # Confirmed present in isolated ground truth
    assert "dominant_label" in labels_df.columns
    assert "has_attack" in labels_df.columns
    assert "attack_flow_ratio" in labels_df.columns
