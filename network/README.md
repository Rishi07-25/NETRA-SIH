# Network Engineering & Traffic Simulation

## Overview
The `network/` directory encapsulates all tools required to simulate realistic enterprise network traffic, inject multi-stage attack scenarios, capture network packets, and parse PCAPs into structured flow records.

## Directory Structure
- `traffic_generator/`: Scripts for synthetic generation of normal and adversarial network traffic.
- `packet_capture/`: Packet sniffing interfaces and PCAP flow feature parsing routines.
- `scenarios/`: Structured scenario configuration profiles (JSON) defining simulated attack stages.
- `pcaps/`: Directory for storing local `.pcap` and `.pcapng` capture files (Git-ignored).

## Primary Owner
**Workstream 3: Network Security**
