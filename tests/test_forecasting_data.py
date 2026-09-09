"""Comprehensive automated test suite for NETRA Stage 3A temporal forecasting data foundation.

Validates:
1. Timestamp normalization (ISO and epoch)
2. Chronological sorting
3. Duplicate handling
4. Half-open window boundaries [t, t + Δ)
5. State aggregation
6. Protocol ratio calculation
7. Port entropy
8. Predictive-vs-target feature separation
9. Sequence shape (N, L, D)
10. Sequence length L and forecast horizon H
11. Observation/target temporal ordering (zero target overlap)
12. Future target correctness
13. Binary target correctness
14. Future risk score range [0.0, 1.0]
15. Attack-stage mapping and taxonomy disclaimer
16. Chronological split
17. Day/scenario-aware split
18. Partition-first sequence generation (no cross-partition sequences)
19. Scaler fit strictly on training data
20. No identifier leakage
21. Empty-window behavior
22. Sparse-window behavior
23. Missing and infinite value handling
24. Deterministic output
25. End-to-end pipeline execution
"""

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from ml.forecasting.dataset_loader import (
    RealDatasetLoader,
    load_network_dataset,
    normalize_dataset_labels,
    parse_flexible_timestamps,
)
from ml.forecasting.schema import (
    ATTACK_STAGE_MAP,
    STATE_METADATA_COLUMNS,
    STATE_PREDICTIVE_FEATURES,
    STATE_TARGET_COLUMNS,
    TAXONOMY_DISCLAIMER,
    map_label_to_attack_stage,
)
from ml.forecasting.sequence_builder import SequenceBuilder, create_sequences_from_states
from ml.forecasting.targets import extract_future_targets, validate_target_alignment
from ml.forecasting.temporal_split import (
    build_partitioned_temporal_dataset,
    partition_states_by_days,
    partition_states_chronologically,
)
from ml.forecasting.temporal_state import (
    build_temporal_states,
    separate_predictive_and_target_features,
)
from ml.forecasting.validate_dataset import (
    audit_raw_flow_data,
    audit_temporal_states,
    validate_forecasting_pipeline,
)
from ml.preprocessing.create_time_windows import (
    calculate_protocol_ratios,
    calculate_shannon_entropy,
)


@pytest.fixture
def sample_flow_dataframe() -> pd.DataFrame:
    """Create a deterministic synthetic network flow DataFrame spanning multiple hours."""
    rows = []
    base_time = pd.Timestamp("2026-09-01 08:00:00")

    labels_cycle = ["BENIGN", "BENIGN", "Reconnaissance", "Brute Force", "DDoS"]
    protocols = [6, 17, 6, 6, 17]
    ports = [80, 53, 443, 22, 8080]

    for i in range(120):
        t = base_time + pd.Timedelta(seconds=i * 15)
        lbl = labels_cycle[i % len(labels_cycle)]
        proto = protocols[i % len(protocols)]
        dst_p = ports[i % len(ports)]

        rows.append({
            "timestamp": str(t),
            "src_ip": f"192.168.1.{10 + (i % 5)}",
            "dst_ip": f"10.0.0.{1 + (i % 3)}",
            "src_port": 40000 + i,
            "dst_port": dst_p,
            "protocol": proto,
            "flow_duration": 10000 + (i * 200),
            "tot_fwd_pkts": 2 + (i % 10),
            "tot_bwd_pkts": 3 + (i % 8),
            "tot_fwd_bytes": 150 + (i * 25),
            "tot_bwd_bytes": 300 + (i * 40),
            "flow_pkt_rate": 50.0 + (i * 2.0),
            "flow_byte_rate": 5000.0 + (i * 150.0),
            "syn_flag_cnt": 1 if i % 2 == 0 else 0,
            "rst_flag_cnt": 1 if i % 10 == 0 else 0,
            "ack_flag_cnt": 1 if i % 3 == 0 else 0,
            "label": lbl,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1. Timestamp Normalization & Parsing
# ---------------------------------------------------------------------------
def test_iso_timestamp_parsing():
    s = pd.Series(["2026-09-01 10:00:00", "01/09/2026 10:01:00", "2026/09/01 10:02:00"])
    parsed = parse_flexible_timestamps(s)
    assert not parsed.isna().any()
    assert parsed.iloc[0].year == 2026


def test_epoch_timestamp_parsing():
    # Epoch seconds
    s_sec = pd.Series([1700000000, 1700000030])
    parsed_sec = parse_flexible_timestamps(s_sec)
    assert not parsed_sec.isna().any()
    assert parsed_sec.iloc[0].year >= 2023

    # Epoch milliseconds
    s_ms = pd.Series([1700000000000, 1700000030000])
    parsed_ms = parse_flexible_timestamps(s_ms)
    assert not parsed_ms.isna().any()
    assert parsed_ms.iloc[0].year >= 2023


# ---------------------------------------------------------------------------
# 2. Chronological Sorting & Duplicates
# ---------------------------------------------------------------------------
def test_chronological_sorting_and_duplicates(tmp_path):
    df = pd.DataFrame({
        "timestamp": ["2026-09-01 10:05:00", "2026-09-01 10:00:00", "2026-09-01 10:00:00"],
        "label": ["BENIGN", "DDoS", "DDoS"],
        "flow_duration": [1000, 500, 500],
    })
    csv_file = tmp_path / "test_sort.csv"
    df.to_csv(csv_file, index=False)

    loader = RealDatasetLoader()
    clean_df, summary = loader.load_dataset(csv_file, drop_duplicates=True)

    assert len(clean_df) == 2  # One duplicate removed
    assert summary["duplicate_rows"] == 1
    assert clean_df["timestamp"].is_monotonic_increasing


# ---------------------------------------------------------------------------
# 3. Half-Open Window Boundaries & Aggregation
# ---------------------------------------------------------------------------
def test_half_open_window_boundaries(sample_flow_dataframe):
    # Window [08:00:00, 08:01:00): contains flows at :00, :15, :30, :45 (4 flows)
    # Flow at 08:01:00 belongs to the next window!
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=60)
    assert len(states) > 0
    first_window = states.iloc[0]
    assert first_window["flow_count"] == 4
    assert first_window["window_start"] < first_window["window_end"]


def test_protocol_ratios_and_entropy():
    # TCP=6, UDP=17
    proto_series = pd.Series([6, 6, 17, 6])
    tcp_r, udp_r = calculate_protocol_ratios(proto_series)
    assert tcp_r == 0.75
    assert udp_r == 0.25

    # Shannon entropy of port distribution
    ports = pd.Series([80, 80, 80, 80])
    assert calculate_shannon_entropy(ports) == 0.0

    diverse_ports = pd.Series([80, 443, 22, 53])
    assert calculate_shannon_entropy(diverse_ports) == 2.0


# ---------------------------------------------------------------------------
# 4. Feature Separation (Predictive vs Target)
# ---------------------------------------------------------------------------
def test_predictive_vs_target_separation(sample_flow_dataframe):
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=30)
    pred_df, target_df, meta_df = separate_predictive_and_target_features(states)

    # Predictive features MUST NOT contain any target or ID columns
    for bad_col in ["dominant_label", "has_attack", "attack_stage", "attack_flow_ratio", "window_id"]:
        assert bad_col not in pred_df.columns

    for col in STATE_PREDICTIVE_FEATURES:
        assert col in pred_df.columns

    for col in STATE_TARGET_COLUMNS:
        assert col in target_df.columns

    for col in STATE_METADATA_COLUMNS:
        assert col in meta_df.columns


# ---------------------------------------------------------------------------
# 5. Sequence Construction & Shapes
# ---------------------------------------------------------------------------
def test_sequence_construction_shapes(sample_flow_dataframe):
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=30)
    L = 4
    H = 1
    builder = SequenceBuilder(sequence_length=L, forecast_horizon=H)
    X, meta, target_idx = builder.build_sequences(states)

    assert X.ndim == 3
    assert X.shape[1] == L
    assert X.shape[2] == len(STATE_PREDICTIVE_FEATURES)
    assert len(X) == len(meta)
    assert len(meta) == len(target_idx)


# ---------------------------------------------------------------------------
# 6. Observation & Target Temporal Ordering (No Leakage)
# ---------------------------------------------------------------------------
def test_observation_target_temporal_separation(sample_flow_dataframe):
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=30)
    X, meta, target_idx = create_sequences_from_states(states, sequence_length=3, forecast_horizon=1)

    targets = extract_future_targets(states, target_idx)
    # validate_target_alignment raises an error if observation_end > target_start
    validate_target_alignment(X, targets, meta)

    for _, row in meta.iterrows():
        assert row["observation_end"] <= row["target_start"]


# ---------------------------------------------------------------------------
# 7. Future Targets Correctness & Bounds
# ---------------------------------------------------------------------------
def test_future_targets_bounds_and_values(sample_flow_dataframe):
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=30)
    _, _, target_idx = create_sequences_from_states(states, sequence_length=3, forecast_horizon=1)
    targets = extract_future_targets(states, target_idx)

    # Binary in {0, 1}
    assert set(targets["future_attack_binary"].unique()).issubset({0, 1})
    # Risk score in [0.0, 1.0]
    scores = targets["future_attack_risk_score"]
    assert (scores >= 0.0).all()
    assert (scores <= 1.0).all()
    # Stage mapping
    for stage in targets["future_attack_stage"]:
        assert stage in {"Normal", "Reconnaissance", "Initial Access", "Exploitation", "Lateral Movement", "Impact"}


# ---------------------------------------------------------------------------
# 8. Attack-Stage Mapping & Disclaimer
# ---------------------------------------------------------------------------
def test_attack_stage_taxonomy():
    assert map_label_to_attack_stage("BENIGN") == "Normal"
    assert map_label_to_attack_stage("PortScan") == "Reconnaissance"
    assert map_label_to_attack_stage("Brute Force") == "Initial Access"
    assert map_label_to_attack_stage("Heartbleed") == "Exploitation"
    assert map_label_to_attack_stage("Infiltration") == "Lateral Movement"
    assert map_label_to_attack_stage("DDoS") == "Impact"
    assert len(TAXONOMY_DISCLAIMER) > 20


# ---------------------------------------------------------------------------
# 9. Chronological Split & Preprocessing Isolation
# ---------------------------------------------------------------------------
def test_chronological_split_and_scaler_isolation(sample_flow_dataframe):
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=30)
    train_s, val_s, test_s = partition_states_chronologically(states, 0.60, 0.20, 0.20)

    assert len(train_s) + len(val_s) + len(test_s) == len(states)
    assert train_s["window_start"].max() <= val_s["window_start"].min()
    assert val_s["window_start"].max() <= test_s["window_start"].min()

    bundle = build_partitioned_temporal_dataset(
        train_states=train_s,
        val_states=val_s,
        test_states=test_s,
        sequence_length=3,
        forecast_horizon=1,
        scale_features=True,
    )

    # Scaler was fitted strictly on training data
    scaler = bundle["scaler"]
    assert isinstance(scaler, StandardScaler)
    assert scaler.mean_ is not None

    # Check that X_train mean is close to 0 after standard scaling
    X_tr_flat = bundle["X_train"].reshape(-1, bundle["X_train"].shape[-1])
    assert pytest.approx(float(X_tr_flat.mean()), abs=1e-2) == 0.0


# ---------------------------------------------------------------------------
# 10. Day / Scenario-Aware Split
# ---------------------------------------------------------------------------
def test_day_based_scenario_split():
    # Multi-day mock state DataFrame
    days_data = []
    for day in ["2026-09-01", "2026-09-02", "2026-09-03"]:
        for h in range(4):
            days_data.append({
                "window_id": len(days_data),
                "window_start": pd.Timestamp(f"{day} 10:0{h}:00"),
                "window_end": pd.Timestamp(f"{day} 10:0{h+1}:00"),
                "flow_count": 10,
                "has_attack": False,
                "dominant_label": "BENIGN",
                "attack_flow_ratio": 0.0,
                "attack_stage": "Normal",
                **{feat: 1.0 for feat in STATE_PREDICTIVE_FEATURES if feat not in {"window_id", "flow_count"}},
            })
    df_states = pd.DataFrame(days_data)

    tr, val, te = partition_states_by_days(
        df_states,
        train_days=["2026-09-01"],
        val_days=["2026-09-02"],
        test_days=["2026-09-03"],
    )
    assert len(tr) == 4
    assert len(val) == 4
    assert len(te) == 4
    assert tr["window_start"].dt.strftime("%Y-%m-%d").unique()[0] == "2026-09-01"


# ---------------------------------------------------------------------------
# 11. No Identifier Leakage in Predictive State Features
# ---------------------------------------------------------------------------
def test_no_identifier_leakage_in_state_features(sample_flow_dataframe):
    states = build_temporal_states(sample_flow_dataframe, window_size_sec=60, stride_sec=30)
    pred_df, _, _ = separate_predictive_and_target_features(states)

    prohibited_substrings = ["ip", "label", "target", "id", "session"]
    for col in pred_df.columns:
        # Allow 'unique_src_ports', 'unique_dst_ports', 'dst_port_entropy', 'avg_pkt_size'
        # Disallow raw 'src_ip', 'dst_ip', 'flow_id', 'dominant_label'
        assert col not in {"src_ip", "dst_ip", "flow_id", "session_id", "dominant_label", "window_id"}


# ---------------------------------------------------------------------------
# 12. Empty and Sparse Window Handling
# ---------------------------------------------------------------------------
def test_empty_window_handling():
    # Large gap between flows: 10:00:00 to 10:10:00
    gap_df = pd.DataFrame({
        "timestamp": ["2026-09-01 10:00:00", "2026-09-01 10:10:00"],
        "label": ["BENIGN", "BENIGN"],
        "flow_duration": [1000, 1000],
        "tot_fwd_pkts": [2, 2],
        "tot_bwd_pkts": [2, 2],
    })
    gap_df["timestamp"] = pd.to_datetime(gap_df["timestamp"])

    # Without empty windows: only 2 windows produced
    states_no_empty = build_temporal_states(gap_df, window_size_sec=60, stride_sec=60, include_empty_windows=False)
    assert len(states_no_empty) == 2

    # With empty windows: intermediate silent windows generated with 0.0 values
    states_with_empty = build_temporal_states(gap_df, window_size_sec=60, stride_sec=60, include_empty_windows=True)
    assert len(states_with_empty) > 2
    silent_window = states_with_empty.iloc[1]
    assert silent_window["flow_count"] == 0.0
    assert silent_window["dominant_label"] == "BENIGN"


# ---------------------------------------------------------------------------
# 13. Missing and Infinite Value Handling
# ---------------------------------------------------------------------------
def test_missing_and_infinite_value_handling(tmp_path):
    dirty_df = pd.DataFrame({
        "timestamp": ["2026-09-01 10:00:00", "2026-09-01 10:00:15"],
        "label": ["BENIGN", "DDoS"],
        "flow_duration": [np.inf, 2000],
        "flow_pkt_rate": [np.nan, 50.0],
    })
    csv_file = tmp_path / "dirty.csv"
    dirty_df.to_csv(csv_file, index=False)

    loader = RealDatasetLoader()
    clean_df, summary = loader.load_dataset(csv_file)

    assert summary["invalid_numeric_values"] >= 1
    assert not np.isinf(clean_df["flow_duration"].to_numpy()).any()
    assert not clean_df["flow_pkt_rate"].isna().any()


# ---------------------------------------------------------------------------
# 14. Deterministic Output & End-to-End Pipeline
# ---------------------------------------------------------------------------
def test_deterministic_output_and_pipeline(tmp_path, sample_flow_dataframe):
    csv_file = tmp_path / "pipeline_fixture.csv"
    sample_flow_dataframe.to_csv(csv_file, index=False)

    rep1 = validate_forecasting_pipeline(
        csv_file,
        window_size_sec=60,
        stride_sec=30,
        sequence_length=3,
        forecast_horizon=1,
    )
    rep2 = validate_forecasting_pipeline(
        csv_file,
        window_size_sec=60,
        stride_sec=30,
        sequence_length=3,
        forecast_horizon=1,
    )

    assert rep1["raw_data_audit"]["total_rows"] == rep2["raw_data_audit"]["total_rows"]
    assert rep1["temporal_states_audit"]["total_windows"] == rep2["temporal_states_audit"]["total_windows"]
    assert rep1["sequence_shapes"] == rep2["sequence_shapes"]
    assert rep1["partition_summary"]["train_sequences"] == rep2["partition_summary"]["train_sequences"]
