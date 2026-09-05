# Packet Capture & Ingestion

## Overview
This module handles network interface tapping, live packet sniffing, and offline PCAP file parsing into structured bidirectional flow objects.

## Components
- `capture.py`: Live socket/interface listener collecting raw network frames.
- `parser.py`: Flow reassembler extracting transport and statistical properties from captured packets.

## Primary Owner
**Workstream 3: Network Security**
