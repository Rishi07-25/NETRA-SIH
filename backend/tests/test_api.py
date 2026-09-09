"""Automated unit and integration tests for NETRA FastAPI backend services and routes.

Validates:
1. Root health check endpoint (status, capabilities, version)
2. /predict endpoint with valid flow features (classification, confidence, top features)
3. /predict endpoint rejection of malformed or missing features (422 status)
4. /anomaly endpoint with flow telemetry (is_anomalous, scores)
5. /forecast endpoint with multi-window chronological sequence (horizons, trajectory)
6. /forecast endpoint rejection of empty history (422 status)
7. /network/stats endpoint (packet throughput, bandwidth, active flows)
8. /events endpoint (alert feed structure)
9. Both root and /api/v1/ prefix routing parity
"""

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)

SAMPLE_FLOW_FEATURES = {
    "dst_port": 80.0,
    "protocol": 6.0,
    "flow_duration": 120000.0,
    "tot_fwd_pkts": 5.0,
    "tot_bwd_pkts": 6.0,
    "tot_fwd_bytes": 450.0,
    "tot_bwd_bytes": 1200.0,
    "flow_pkt_rate": 91.67,
    "flow_byte_rate": 13750.0,
    "syn_flag_cnt": 1.0,
    "rst_flag_cnt": 0.0,
    "ack_flag_cnt": 5.0,
    "fwd_bwd_pkt_ratio": 0.833,
    "fwd_bwd_byte_ratio": 0.375,
    "avg_pkt_size": 150.0,
}

SAMPLE_TEMPORAL_WINDOW = {
    "flow_count": 10.0,
    "tot_fwd_pkts": 50.0,
    "tot_bwd_pkts": 45.0,
    "tot_fwd_bytes": 5000.0,
    "tot_bwd_bytes": 12000.0,
    "avg_pkt_size": 180.0,
    "flow_duration": 45000.0,
    "flow_pkt_rate": 120.0,
    "flow_byte_rate": 15000.0,
    "syn_flag_cnt": 2.0,
    "rst_flag_cnt": 0.0,
    "ack_flag_cnt": 40.0,
    "fwd_bwd_pkt_ratio": 1.11,
    "fwd_bwd_byte_ratio": 0.42,
    "unique_src_ports": 8.0,
    "unique_dst_ports": 3.0,
    "dst_port_entropy": 1.25,
    "tcp_ratio": 0.85,
    "udp_ratio": 0.15,
}


def test_health_check_endpoint():
    """Verify GET / returns online status and registered capabilities."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "capabilities" in data
    assert "attack_classification" in data["capabilities"]


def test_predict_endpoint_valid_payload():
    """Verify POST /predict returns valid classification response."""
    payload = {
        "flow_id": "test-flow-001",
        "features": SAMPLE_FLOW_FEATURES,
    }
    response = client.post("/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "attack_class" in data
    assert isinstance(data["attack_class"], str)
    assert 0.0 <= data["confidence"] <= 1.0
    assert isinstance(data["top_features"], list)


def test_predict_endpoint_api_v1_parity():
    """Verify POST /api/v1/predict behaves identically to /predict/."""
    payload = {"features": SAMPLE_FLOW_FEATURES}
    response = client.post("/api/v1/predict/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "attack_class" in data


def test_predict_endpoint_missing_feature_validation():
    """Verify POST /predict returns 422 on incomplete feature vector."""
    incomplete_features = {"dst_port": 80.0}
    response = client.post("/predict/", json={"features": incomplete_features})
    assert response.status_code == 422


def test_anomaly_endpoint_valid_payload():
    """Verify POST /anomaly returns behavioral outlier scoring."""
    payload = {"features": SAMPLE_FLOW_FEATURES}
    response = client.post("/anomaly/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "is_anomalous" in data
    assert isinstance(data["is_anomalous"], bool)
    assert 0.0 <= data["anomaly_score"] <= 1.0
    assert "raw_score" in data


def test_forecast_endpoint_valid_sequence():
    """Verify POST /forecast returns multi-horizon risk projections."""
    # Sequence of 5 historical windows
    history = [SAMPLE_TEMPORAL_WINDOW.copy() for _ in range(5)]
    payload = {
        "window_count": len(history),
        "history": history,
    }
    response = client.post("/forecast/", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "current_threat_score" in data
    assert "forecasts" in data
    assert len(data["forecasts"]) >= 3
    assert data["trajectory"] in {"STABLE", "ESCALATING", "DECLINING"}
    for f in data["forecasts"]:
        assert "horizon" in f
        assert 0.0 <= f["risk_score"] <= 100.0
        assert 0.0 <= f["attack_probability"] <= 1.0


def test_forecast_endpoint_empty_history_validation():
    """Verify POST /forecast rejects empty history."""
    response = client.post("/forecast/", json={"window_count": 0, "history": []})
    assert response.status_code == 422


def test_network_stats_endpoint():
    """Verify GET /network/stats returns real-time throughput metrics."""
    response = client.get("/network/stats")
    assert response.status_code == 200
    data = response.json()
    assert "active_flows" in data
    assert data["active_flows"] > 0
    assert data["packets_per_second"] >= 0.0
    assert data["bytes_per_second"] >= 0.0
    assert 0.0 <= data["anomaly_rate"] <= 1.0


def test_security_events_endpoint():
    """Verify GET /events returns security alert stream."""
    response = client.get("/events/")
    assert response.status_code == 200
    data = response.json()
    assert "total_events" in data
    assert isinstance(data["events"], list)
    assert data["total_events"] == len(data["events"])
    if data["events"]:
        evt = data["events"][0]
        assert "event_id" in evt
        assert "severity" in evt
        assert "title" in evt
