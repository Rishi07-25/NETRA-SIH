"""Service managing network traffic telemetry, metrics aggregation, and event history."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd

logger = logging.getLogger(__name__)

SAMPLE_FLOWS_PATH = "data/samples/synthetic_network_flows.csv"


class TrafficService:
    """Service orchestrator for traffic telemetry stream and window buffers."""

    def __init__(self, sample_flows_path: str = SAMPLE_FLOWS_PATH):
        self.sample_flows_path = Path(sample_flows_path)
        self.events_log: List[Dict[str, Any]] = [
            {
                "event_id": "EVT-001",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "PRECURSOR_DETECTED",
                "severity": "MEDIUM",
                "title": "Stealth Port Sweep Detected",
                "description": "Sequential SYN scanning observed across destination ports 21-25.",
                "details": {"source_ip": "172.16.0.50", "protocol": "TCP", "probed_ports": [21, 22, 23, 25]},
            },
            {
                "event_id": "EVT-002",
                "timestamp": datetime.now(timezone.utc),
                "event_type": "THREAT_ESCALATING",
                "severity": "HIGH",
                "title": "Imminent Credential Brute-Force Risk",
                "description": "5-minute horizon forecast projects elevated credential stuffing on port 22.",
                "details": {"target_port": 22, "forecast_probability": 0.74, "risk_score": 68.5},
            }
        ]

    def get_current_metrics(self) -> Dict[str, Any]:
        """Retrieve aggregated throughput and flow metrics."""
        if self.sample_flows_path.exists():
            try:
                df = pd.read_csv(self.sample_flows_path)
                active_flows = int(len(df))
                mean_pps = float(df["flow_pkt_rate"].mean()) if "flow_pkt_rate" in df.columns else 1250.0
                mean_bps = float(df["flow_byte_rate"].mean()) if "flow_byte_rate" in df.columns else 64000.0
                attack_count = int((df["label"] != "BENIGN").sum()) if "label" in df.columns else 0
                anomaly_rate = float(attack_count / active_flows) if active_flows > 0 else 0.0
                
                return {
                    "active_flows": active_flows,
                    "packets_per_second": round(mean_pps, 2),
                    "bytes_per_second": round(mean_bps, 2),
                    "anomaly_rate": round(anomaly_rate, 4),
                }
            except Exception as e:
                logger.warning(f"Error reading flow samples: {e}")

        return {
            "active_flows": 142,
            "packets_per_second": 1250.0,
            "bytes_per_second": 64000.0,
            "anomaly_rate": 0.12,
        }

    def get_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Retrieve recent security and early-warning events."""
        return self.events_log[:limit]

    def record_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Append an event to the security audit feed."""
        if "timestamp" not in event:
            event["timestamp"] = datetime.now(timezone.utc)
        if "event_id" not in event:
            event["event_id"] = f"EVT-{len(self.events_log) + 1:03d}"
        self.events_log.insert(0, event)
        return event
