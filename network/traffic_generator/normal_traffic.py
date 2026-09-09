"""Generator for benign enterprise network traffic patterns."""

from datetime import datetime, timezone
import random
from typing import Any, Dict, List, Optional


def generate_normal_traffic(
    target_host: str = "10.0.0.1",
    duration_sec: int = 60,
    flow_count: int = 10,
    random_seed: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Simulate benign enterprise network communication flows.

    Returns structured flow records adhering to canonical NETRA flow schemas.
    """
    if random_seed is not None:
        random.seed(random_seed)

    flows = []
    base_time = datetime.now(timezone.utc)

    for i in range(flow_count):
        # Normal web (80, 443) or DNS (53) traffic
        dst_port = random.choice([80, 443, 53, 8080])
        proto = 17 if dst_port == 53 else 6
        flow_duration = random.randint(15000, 250000)
        tot_fwd_pkts = random.randint(2, 20)
        tot_bwd_pkts = random.randint(2, 25)
        tot_fwd_bytes = tot_fwd_pkts * random.randint(50, 200)
        tot_bwd_bytes = tot_bwd_pkts * random.randint(100, 1400)
        dur_sec = max(flow_duration / 1000000.0, 0.001)

        flows.append({
            "timestamp": base_time.isoformat(),
            "src_ip": f"192.168.1.{random.randint(10, 50)}",
            "src_port": random.randint(49152, 65535),
            "dst_ip": target_host,
            "dst_port": dst_port,
            "protocol": proto,
            "flow_duration": flow_duration,
            "tot_fwd_pkts": tot_fwd_pkts,
            "tot_bwd_pkts": tot_bwd_pkts,
            "tot_fwd_bytes": tot_fwd_bytes,
            "tot_bwd_bytes": tot_bwd_bytes,
            "flow_pkt_rate": round((tot_fwd_pkts + tot_bwd_pkts) / dur_sec, 2),
            "flow_byte_rate": round((tot_fwd_bytes + tot_bwd_bytes) / dur_sec, 2),
            "syn_flag_cnt": 1 if proto == 6 else 0,
            "rst_flag_cnt": 0,
            "ack_flag_cnt": tot_bwd_pkts if proto == 6 else 0,
            "label": "BENIGN",
        })

    return flows
