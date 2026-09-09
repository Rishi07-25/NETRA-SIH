"""Simulation script for Denial of Service and distributed volumetric floods."""

from datetime import datetime, timezone
import random
from typing import Any, Dict, List, Optional


def simulate_ddos(
    target_host: str = "10.0.0.1",
    target_port: int = 80,
    flow_count: int = 20,
    packet_rate: int = 5000,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simulate volumetric flood patterns over target IP endpoints."""
    if random_seed is not None:
        random.seed(random_seed)

    flows = []
    base_time = datetime.now(timezone.utc)

    for i in range(flow_count):
        # Volumetric flood: high packet rate, small packet size, predominantly forward packets
        flow_duration = random.randint(1000, 20000)
        dur_sec = max(flow_duration / 1000000.0, 0.0001)
        tot_fwd_pkts = random.randint(50, 300)
        tot_bwd_pkts = random.randint(0, 5)
        tot_fwd_bytes = tot_fwd_pkts * 64
        tot_bwd_bytes = tot_bwd_pkts * 40

        flows.append({
            "timestamp": base_time.isoformat(),
            "src_ip": f"198.51.100.{random.randint(1, 254)}",
            "src_port": random.randint(1024, 65535),
            "dst_ip": target_host,
            "dst_port": target_port,
            "protocol": 6,
            "flow_duration": flow_duration,
            "tot_fwd_pkts": tot_fwd_pkts,
            "tot_bwd_pkts": tot_bwd_pkts,
            "tot_fwd_bytes": tot_fwd_bytes,
            "tot_bwd_bytes": tot_bwd_bytes,
            "flow_pkt_rate": round(tot_fwd_pkts / dur_sec, 2),
            "flow_byte_rate": round(tot_fwd_bytes / dur_sec, 2),
            "syn_flag_cnt": tot_fwd_pkts,
            "rst_flag_cnt": 0,
            "ack_flag_cnt": tot_bwd_pkts,
            "label": "DDoS",
        })

    return flows
