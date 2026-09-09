# Network Engineering & Traffic Simulation

## Overview
The `network/` directory encapsulates all tools required to simulate realistic enterprise network traffic, inject multi-stage attack scenarios, capture network packets, and parse PCAPs into structured flow records.

## Components
- `traffic_generator/`:
  - `normal_traffic.py`: Simulates benign enterprise network communication flows (HTTP/HTTPS/DNS).
  - `scan_simulation.py`: Simulates reconnaissance sweeps across IP ranges and target ports.
  - `brute_force_simulation.py`: Simulates credential brute-force and SSH dictionary attacks.
  - `ddos_simulation.py`: Simulates volumetric traffic floods (SYN floods).
- `packet_capture/`:
  - `capture.py`: Live socket/interface listener and synthetic standards-compliant PCAP generator.
  - `parser.py`: Pure-Python bidirectional flow extractor reconstructing sessions into NETRA flow format.
- `scenarios/`: Structured scenario configuration profiles (JSON) defining simulated attack stages (`normal.json`, `reconnaissance.json`, `escalation.json`, `ddos.json`).
- `pcaps/`: Directory for storing local `.pcap` and `.pcapng` capture files (Git-ignored).

## Usage & Execution
```bash
# Run automated network components test suite
pytest tests/test_network_components.py
```
