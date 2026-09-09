"""Automated tests for network traffic generation and packet capture modules."""

from pathlib import Path
import pytest

from network.packet_capture.capture import start_packet_capture, write_synthetic_pcap
from network.packet_capture.parser import parse_pcap_to_flows
from network.traffic_generator.brute_force_simulation import simulate_brute_force
from network.traffic_generator.ddos_simulation import simulate_ddos
from network.traffic_generator.normal_traffic import generate_normal_traffic
from network.traffic_generator.scan_simulation import simulate_port_scan


def test_generate_normal_traffic():
    """Verify normal traffic generator produces structured flow records."""
    flows = generate_normal_traffic(target_host="10.0.0.1", flow_count=5, random_seed=42)
    assert len(flows) == 5
    for f in flows:
        assert f["label"] == "BENIGN"
        assert f["protocol"] in (6, 17)
        assert f["flow_duration"] > 0
        assert f["tot_fwd_pkts"] > 0


def test_simulate_port_scan():
    """Verify port scan simulator produces reconnaissance attack labels."""
    flows = simulate_port_scan(target_subnet="10.0.0.1", intensity="stealth", random_seed=42)
    assert len(flows) > 0
    for f in flows:
        assert f["label"] == "Reconnaissance"
        assert f["syn_flag_cnt"] == 1
        assert f["rst_flag_cnt"] == 1


def test_simulate_brute_force():
    """Verify brute force simulator produces authentication attack flows."""
    flows = simulate_brute_force(target_service="10.0.0.1", attempts=8, random_seed=42)
    assert len(flows) == 8
    for f in flows:
        assert f["label"] == "Brute Force"
        assert f["dst_port"] == 22


def test_simulate_ddos():
    """Verify DDoS simulation produces high packet volume flows."""
    flows = simulate_ddos(target_host="10.0.0.1", flow_count=10, random_seed=42)
    assert len(flows) == 10
    for f in flows:
        assert f["label"] == "DDoS"
        assert f["tot_fwd_pkts"] >= 50


def test_pcap_generation_and_parsing(tmp_path):
    """Verify writing a PCAP and parsing it back into structured bidirectional flows."""
    pcap_file = tmp_path / "test.pcap"
    created_path = write_synthetic_pcap(str(pcap_file), packet_count=12)
    assert Path(created_path).exists()

    flows = parse_pcap_to_flows(created_path)
    assert len(flows) > 0
    flow = flows[0]
    assert "src_ip" in flow
    assert "dst_ip" in flow
    assert "flow_duration" in flow
    assert "tot_fwd_pkts" in flow
    assert flow["tot_fwd_pkts"] > 0
