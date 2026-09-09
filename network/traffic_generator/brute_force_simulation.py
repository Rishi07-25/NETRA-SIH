"""Simulation script for credential stuffing and brute-force attacks."""

from datetime import datetime, timezone
import random
from typing import Any, Dict, List, Optional


def simulate_brute_force(
    target_service: str = "10.0.0.1",
    target_port: int = 22,
    attempts: int = 15,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simulate rapid failed authentication attempts."""
    if random_seed is not None:
        random.seed(random_seed)

    flows = []
    base_time = datetime.now(timezone.utc)
    attacker_ip = "172.16.0.60"

    for _ in range(attempts):
        # SSH Brute force characteristic: 24,000 - 30,000 us duration, ~11-15 fwd pkts, ~13-16 bwd pkts
        flow_duration = random.randint(24000, 30000)
        dur_sec = flow_duration / 1000000.0
        tot_fwd_pkts = random.randint(11, 15)
        tot_bwd_pkts = random.randint(13, 16)
        tot_fwd_bytes = random.randint(900, 1150)
        tot_bwd_bytes = random.randint(1350, 1650)

        flows.append({
            "timestamp": base_time.isoformat(),
            "src_ip": attacker_ip,
            "src_port": random.randint(42000, 42100),
            "dst_ip": target_service,
            "dst_port": target_port,
            "protocol": 6,
            "flow_duration": flow_duration,
            "tot_fwd_pkts": tot_fwd_pkts,
            "tot_bwd_pkts": tot_bwd_pkts,
            "tot_fwd_bytes": tot_fwd_bytes,
            "tot_bwd_bytes": tot_bwd_bytes,
            "flow_pkt_rate": round((tot_fwd_pkts + tot_bwd_pkts) / dur_sec, 2),
            "flow_byte_rate": round((tot_fwd_bytes + tot_bwd_bytes) / dur_sec, 2),
            "syn_flag_cnt": 1,
            "rst_flag_cnt": 0,
            "ack_flag_cnt": tot_fwd_pkts,
            "label": "Brute Force",
        })

    return flows
