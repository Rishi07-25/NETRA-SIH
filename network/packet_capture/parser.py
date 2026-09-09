"""PCAP and packet parser for bidirectional flow extraction.

Safely parses pcap metadata into structured flow records compatible with NETRA pipelines.
Provides a pure Python parser without requiring privileged network packet drivers.
"""

from datetime import datetime, timezone
from pathlib import Path
import struct
from typing import Any, Dict, List


def parse_pcap_to_flows(pcap_path: str) -> List[Dict[str, Any]]:
    """Parse raw PCAP file into structured flow telemetry records.

    Reads global PCAP header (magic 0xa1b2c3d4 or 0xd4c3b2a1) and per-packet headers,
    aggregating packets into bidirectional 5-tuple flow records.
    """
    path = Path(pcap_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP file not found: {path}")

    # Standard libpcap global header is 24 bytes
    with open(path, "rb") as f:
        global_header = f.read(24)
        if len(global_header) < 24:
            raise ValueError(f"PCAP file {pcap_path} is too small or truncated.")

        magic = global_header[:4]
        if magic in (b"\xa1\xb2\xc3\xd4", b"\xd4\xc3\xb2\xa1"):
            endian = "<" if magic == b"\xd4\xc3\xb2\xa1" else ">"
        else:
            raise ValueError(f"Invalid PCAP magic header: {magic.hex()}")

        flows_map: Dict[tuple, Dict[str, Any]] = {}
        pkt_count = 0

        while True:
            pkt_hdr = f.read(16)
            if len(pkt_hdr) < 16:
                break

            ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f"{endian}IIII", pkt_hdr)
            pkt_data = f.read(incl_len)
            if len(pkt_data) < incl_len:
                break

            pkt_count += 1
            ts = ts_sec + (ts_usec / 1e6)

            # Minimal Ethernet + IPv4 parsing (offset 14 for Ethernet II)
            if incl_len >= 34 and pkt_data[12:14] == b"\x08\x00":
                ip_header = pkt_data[14:34]
                proto = ip_header[9]
                src_ip = ".".join(str(b) for b in ip_header[12:16])
                dst_ip = ".".join(str(b) for b in ip_header[16:20])

                src_port = 0
                dst_port = 0
                if proto in (6, 17) and incl_len >= 38:
                    trans_hdr = pkt_data[34:38]
                    src_port, dst_port = struct.unpack(">HH", trans_hdr)

                key = (min(src_ip, dst_ip), max(src_ip, dst_ip), min(src_port, dst_port), max(src_port, dst_port), proto)

                if key not in flows_map:
                    flows_map[key] = {
                        "timestamp": datetime.fromtimestamp(ts, timezone.utc).isoformat(),
                        "src_ip": src_ip,
                        "src_port": src_port,
                        "dst_ip": dst_ip,
                        "dst_port": dst_port,
                        "protocol": proto,
                        "start_time": ts,
                        "end_time": ts,
                        "tot_fwd_pkts": 1,
                        "tot_bwd_pkts": 0,
                        "tot_fwd_bytes": incl_len,
                        "tot_bwd_bytes": 0,
                        "syn_flag_cnt": 0,
                        "rst_flag_cnt": 0,
                        "ack_flag_cnt": 0,
                    }
                else:
                    rec = flows_map[key]
                    rec["end_time"] = ts
                    if rec["src_ip"] == src_ip:
                        rec["tot_fwd_pkts"] += 1
                        rec["tot_fwd_bytes"] += incl_len
                    else:
                        rec["tot_bwd_pkts"] += 1
                        rec["tot_bwd_bytes"] += incl_len

    # Finalize flow durations and rates
    results = []
    for rec in flows_map.values():
        dur_sec = max(rec["end_time"] - rec["start_time"], 0.0001)
        dur_usec = int(dur_sec * 1e6)
        total_pkts = rec["tot_fwd_pkts"] + rec["tot_bwd_pkts"]
        total_bytes = rec["tot_fwd_bytes"] + rec["tot_bwd_bytes"]

        results.append({
            "timestamp": rec["timestamp"],
            "src_ip": rec["src_ip"],
            "src_port": rec["src_port"],
            "dst_ip": rec["dst_ip"],
            "dst_port": rec["dst_port"],
            "protocol": rec["protocol"],
            "flow_duration": dur_usec,
            "tot_fwd_pkts": rec["tot_fwd_pkts"],
            "tot_bwd_pkts": rec["tot_bwd_pkts"],
            "tot_fwd_bytes": rec["tot_fwd_bytes"],
            "tot_bwd_bytes": rec["tot_bwd_bytes"],
            "flow_pkt_rate": round(total_pkts / dur_sec, 2),
            "flow_byte_rate": round(total_bytes / dur_sec, 2),
            "syn_flag_cnt": rec["syn_flag_cnt"],
            "rst_flag_cnt": rec["rst_flag_cnt"],
            "ack_flag_cnt": rec["ack_flag_cnt"],
            "label": "BENIGN",
        })

    return results
