# Cybersecurity Analysis & Threat Modeling

## MITRE ATT&CK Mapping
NETRA aligns its forecasting capability directly with early-stage tactics within the **MITRE ATT&CK for Enterprise** framework:

| Tactic | Techniques Targeted | NETRA Telemetry Indicators |
|---|---|---|
| **Reconnaissance (TA0043)** | Active Scanning (T1595), IP/Port Sweeps | High SYN counts, unbalanced flow ratios, rapid RST bursts |
| **Initial Access (TA0001)** | Exploit Public-Facing Application (T1190) | Unusual payload sizes, unusual port targeting, rapid session teardown |
| **Credential Access (TA0006)** | Brute Force (T1110), Password Spraying | High failed connection frequency, repetitive TCP half-open flows |
| **Impact (TA0040)** | Network Denial of Service (T1498) | Volume spikes, packet per second (pps) rate surges, bandwidth saturation |

---

## The Temporal Precursor Hypothesis
Most devastating network attacks are preceded by predictable, detectable precursor activity:
1. **Phase 1: Footprinting (T - 30m to T - 10m):** Broad, low-intensity reconnaissance scans.
2. **Phase 2: Vulnerability Enumeration (T - 10m to T - 2m):** Targeted probing of open ports and services.
3. **Phase 3: Weaponization / Delivery (T - 2m to T):** Staging high-frequency authentication attempts or exploit payloads.
4. **Phase 4: Impact / Execution (T onwards):** Full-scale breach or volumetric DoS.

NETRA's core security premise is identifying the transition between Phase 1 and Phase 3, empowering defenders to trigger active defensive countermeasures (IP shun, dynamic ACL, rate limiting) *before* Phase 4 occurs.
