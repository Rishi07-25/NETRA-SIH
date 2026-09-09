"""Simulation script for network reconnaissance and port scanning."""

from datetime import datetime, timezone
import random
from typing import Any, Dict, List, Optional


def simulate_port_scan(
    target_subnet: str = "10.0.0.1",
    intensity: str = "stealth",
    scan_ports: Optional[List[int]] = None,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simulate reconnaissance sweeps across IP ranges and target ports."""
    if random_seed is not None:
        random.seed(random_seed)

    ports = scan_ports or [21, 22, 23, 25, 80, 110, 143, 443, 445, 3389]
    flows = []
    base_time = datetime.now(timezone.utc)
    attacker_ip = "172.16.0.50"

    for port in ports:
        # Port scan characteristics: short duration, 1-2 packets, high SYN, RST returned
        flow_duration = random.randint(400, 1500)
        dur_sec = max(flow_duration / 1000000.0, 0.0001)
        tot_fwd_pkts = 1
        tot_bwd_pkts = 0 if intensity == "stealth" else 1
        tot_fwd_bytes = 40
        tot_bwd_bytes = 0

        flows.append({
            "timestamp": base_time.isoformat(),
            "src_ip": attacker_ip,
            "src_port": random.randint(30000, 45000),
            "dst_ip": target_subnet,
            "dst_port": port,
            "protocol": 6,
            "flow_duration": flow_duration,
            "tot_fwd_pkts": tot_fwd_pkts,
            "tot_bwd_pkts": tot_bwd_pkts,
            "tot_fwd_bytes": tot_fwd_bytes,
            "tot_bwd_bytes": tot_bwd_bytes,
            "flow_pkt_rate": round(tot_fwd_pkts / dur_sec, 2),
            "flow_byte_rate": round(tot_fwd_bytes / dur_sec, 2),
            "syn_flag_cnt": 1,
            "rst_flag_cnt": 1,
            "ack_flag_cnt": 0,
            "label": "Reconnaissance",
        })

    return flows
