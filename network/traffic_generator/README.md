# Traffic Generation Engine

## Overview
This module simulates both standard enterprise background traffic and synthetic cyber attack sequences to validate NETRA's real-time detection and forecasting capabilities.

## Components
- `normal_traffic.py`: Generates benign traffic patterns (HTTP/HTTPS browsing, DNS queries, SSH/SMB sessions).
- `scan_simulation.py`: Simulates horizontal/vertical port sweeps and stealthy network discovery.
- `brute_force_simulation.py`: Simulates repetitive credential attacks against protocols like SSH or HTTP basic auth.
- `ddos_simulation.py`: Simulates volumetric traffic floods (SYN floods, UDP amplification).

## Primary Owner
**Workstream 3: Network Security**
