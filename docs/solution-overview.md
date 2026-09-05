# Solution Overview: NETRA

## System Philosophy
**"Don't just detect an attack after it happens — forecast the likelihood of an attack before it happens."**

NETRA bridges the critical gap between reactive detection and proactive risk mitigation. By modeling network communications as continuous time-series flows with evolving statistical and structural properties, NETRA captures subtle preparatory behaviors—such as slow reconnaissance scans, credential stuffing ramps, and distributed horizontal probes—prior to high-volume disruptions or lateral movement.

---

## High-Level Architecture Flow
```text
[ Network Telemetry / PCAP ]
            │
            ▼
[ Flow Extraction & Session Reassembly ]
            │
            ▼
[ Feature Engineering & Time-Window Aggregation ]
            │
      ┌─────┴─────────────────────┐
      ▼                           ▼
[ Behavioral Anomaly Engine ] [ Attack Classification Engine ]
      └─────┬─────────────────────┘
            ▼
[ Temporal Risk Forecasting Engine ]
            │
            ▼
[ Threat Score & Early Warning Computation ]
            │
            ▼
[ FastAPI Backend Dispatcher ]
            │
            ▼
[ Next-Gen SOC Analyst Dashboard ]
```

---

## Operational Workflow
1. **Network Ingestion:** Continuous capture of raw packet data or flow telemetry (NetFlow, IPFIX, Zeek/Bro logs).
2. **Feature Computation:** Aggregation of bi-directional flow features over rolling time windows (e.g., $\Delta t = 10s, 30s, 60s$).
3. **Multi-tiered ML Engine:**
   - *Tier 1:* Unsupervised anomaly detection flags out-of-distribution behaviors.
   - *Tier 2:* Supervised multi-class classification categorizes attack techniques.
   - *Tier 3:* Sequence/time-series forecasting evaluates temporal escalation to forecast impending breach probabilities.
4. **Threat Scoring:** Dynamic Threat Score ($0 - 100$) reflecting threat imminence, attack severity, and asset exposure.
5. **Analyst Presentation:** Interactive visualization on the SOC dashboard with timeline escalation curves and recommended mitigation rules.
