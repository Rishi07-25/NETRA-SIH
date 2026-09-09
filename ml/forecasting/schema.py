"""Canonical schemas, column aliases, and attack-stage taxonomy for NETRA Stage 3A.

Defines feature vocabularies for flow telemetry and temporal network state S_t,
dataset column mappings for CIC-IDS2017 and UNSW-NB15, label canonicalization rules,
and the operational attack-stage taxonomy.
"""

from __future__ import annotations

from typing import Dict, List, Set

# Mandatory Operational Heuristic Disclaimer
TAXONOMY_DISCLAIMER: str = (
    "Attack stage is a NETRA-derived operational taxonomy, "
    "not a ground-truth label directly provided by the dataset."
)

# ---------------------------------------------------------------------------
# 1. Flow-Level Canonical Feature Vocabulary
# ---------------------------------------------------------------------------
CANONICAL_FLOW_NUMERIC_FEATURES: List[str] = [
    "flow_duration",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "tot_fwd_bytes",
    "tot_bwd_bytes",
    "flow_pkt_rate",
    "flow_byte_rate",
    "syn_flag_cnt",
    "rst_flag_cnt",
    "ack_flag_cnt",
    "fwd_bwd_pkt_ratio",
    "fwd_bwd_byte_ratio",
    "avg_pkt_size",
]

FLOW_METADATA_COLUMNS: Set[str] = {
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "label",
}

# ---------------------------------------------------------------------------
# 2. Temporal State S_t Feature Definitions
# ---------------------------------------------------------------------------
# Predictive features constituting S_t vector (input to temporal models)
STATE_PREDICTIVE_FEATURES: List[str] = [
    # Flow Volume Metrics
    "flow_count",
    "tot_fwd_pkts",
    "tot_bwd_pkts",
    "tot_fwd_bytes",
    "tot_bwd_bytes",
    "avg_pkt_size",
    # Rate Dynamics
    "flow_duration",
    "flow_pkt_rate",
    "flow_byte_rate",
    # TCP Control Flag Counts
    "syn_flag_cnt",
    "rst_flag_cnt",
    "ack_flag_cnt",
    # Directional Flow Ratios
    "fwd_bwd_pkt_ratio",
    "fwd_bwd_byte_ratio",
    # Network Endpoint & Semantic Distributions
    "unique_src_ports",
    "unique_dst_ports",
    "dst_port_entropy",
    "tcp_ratio",
    "udp_ratio",
]

# State Window Identifiers (Metadata only - MUST NOT be model inputs)
STATE_METADATA_COLUMNS: List[str] = [
    "window_id",
    "window_start",
    "window_end",
]

# Ground-Truth Target Columns (MUST NOT be predictive model inputs)
STATE_TARGET_COLUMNS: List[str] = [
    "dominant_label",
    "has_attack",
    "attack_flow_ratio",
    "attack_stage",
]

# ---------------------------------------------------------------------------
# 3. Dataset Column Aliases
# ---------------------------------------------------------------------------
# CIC-IDS2017 raw column names (CICFlowMeter) to canonical normalized names
CIC_IDS2017_COLUMN_MAP: Dict[str, str] = {
    # Flow duration & counts
    "flow_duration": "flow_duration",
    "total_fwd_packets": "tot_fwd_pkts",
    "total_backward_packets": "tot_bwd_pkts",
    "total_length_of_fwd_packets": "tot_fwd_bytes",
    "total_length_of_bwd_packets": "tot_bwd_bytes",
    # Rates
    "flow_bytes_s": "flow_byte_rate",
    "flow_packets_s": "flow_pkt_rate",
    # Flags
    "syn_flag_count": "syn_flag_cnt",
    "rst_flag_count": "rst_flag_cnt",
    "ack_flag_count": "ack_flag_cnt",
    # Network identifiers & protocol
    "destination_port": "dst_port",
    "protocol": "protocol",
    "source_ip": "src_ip",
    "destination_ip": "dst_ip",
    "source_port": "src_port",
    "timestamp": "timestamp",
    "label": "label",
}

# UNSW-NB15 raw column mapping for cross-dataset compatibility
UNSW_NB15_COLUMN_MAP: Dict[str, str] = {
    "dur": "flow_duration",
    "spkts": "tot_fwd_pkts",
    "dpkts": "tot_bwd_pkts",
    "sbytes": "tot_fwd_bytes",
    "dbytes": "tot_bwd_bytes",
    "proto": "protocol",
    "sport": "src_port",
    "dsport": "dst_port",
    "srcip": "src_ip",
    "dstip": "dst_ip",
    "stime": "timestamp",
    "attack_cat": "label",
    "label": "has_attack_raw",
}

# ---------------------------------------------------------------------------
# 4. Canonical Label Normalization Dictionary
# ---------------------------------------------------------------------------
LABEL_NORMALIZATION_MAP: Dict[str, str] = {
    # Benign variations
    "benign": "BENIGN",
    "normal": "BENIGN",
    "0": "BENIGN",
    # Reconnaissance / Scans
    "portscan": "Reconnaissance",
    "port_scan": "Reconnaissance",
    "ipsweep": "Reconnaissance",
    "portsweep": "Reconnaissance",
    "reconnaissance": "Reconnaissance",
    "nmap": "Reconnaissance",
    # Brute Force / Credential attacks
    "ssh-bruteforce": "Brute Force",
    "ssh-patator": "Brute Force",
    "ftp-patator": "Brute Force",
    "bruteforce": "Brute Force",
    "brute_force": "Brute Force",
    # DoS Variations (CIC-IDS2017 Wednesday)
    "dos": "DoS",
    "dos slowloris": "DoS",
    "dos slowhttptest": "DoS",
    "dos hulk": "DoS",
    "dos goldeneye": "DoS",
    # DDoS Variations (CIC-IDS2017 Friday)
    "ddos": "DDoS",
    "ddos-synflood": "DDoS",
    # Web Attacks (CIC-IDS2017 Thursday)
    "web attack": "Web Attack",
    "web attack – brute force": "Web Attack",
    "web attack – xss": "Web Attack",
    "web attack – sql injection": "Web Attack",
    "web attack - brute force": "Web Attack",
    "web attack - xss": "Web Attack",
    "web attack - sql injection": "Web Attack",
    # Infiltration / Botnet
    "infiltration": "Infiltration",
    "bot": "Botnet",
    "botnet": "Botnet",
    # Specific Exploitation
    "heartbleed": "Heartbleed",
}

# ---------------------------------------------------------------------------
# 5. NETRA Operational Attack-Stage Taxonomy
# ---------------------------------------------------------------------------
# Maps canonical attack categories to operational tactical progression stages
ATTACK_STAGE_MAP: Dict[str, str] = {
    "BENIGN": "Normal",
    "Reconnaissance": "Reconnaissance",
    "Brute Force": "Initial Access",
    "Web Attack": "Exploitation",
    "Heartbleed": "Exploitation",
    "Infiltration": "Lateral Movement",
    "Botnet": "Lateral Movement",
    "DoS": "Impact",
    "DDoS": "Impact",
}

VALID_ATTACK_STAGES: List[str] = [
    "Normal",
    "Reconnaissance",
    "Initial Access",
    "Exploitation",
    "Lateral Movement",
    "Impact",
]


def map_label_to_attack_stage(label: str) -> str:
    """Map canonical attack label to NETRA operational stage.

    NOTE: Attack stage is a NETRA-derived operational taxonomy,
    not a ground-truth label directly provided by the dataset.
    """
    clean_label = str(label).strip()
    if clean_label in ATTACK_STAGE_MAP:
        return ATTACK_STAGE_MAP[clean_label]

    norm_label = LABEL_NORMALIZATION_MAP.get(clean_label.lower(), clean_label)
    if norm_label in ATTACK_STAGE_MAP:
        return ATTACK_STAGE_MAP[norm_label]

    if norm_label.upper() == "BENIGN" or norm_label.lower() in ("normal", "0"):
        return "Normal"
    return "Exploitation"
