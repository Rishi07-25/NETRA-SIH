# SIH Presentation Speaker Notes

## Pitch Structure (Total: 8 Minutes + 5 Minutes Q&A)

### 1. The Hook & The Problem (Speaker 1 - 0:00 to 1:30)
- "Respected judges, every modern SOC has a massive flaw: it tells you you've been attacked *after* the damage is done."
- Introduce the problem statement: **SIH26153 (NTRO)**.
- Introduce NETRA's paradigm shift: **Network Threat Early-warning & Risk Analytics**.
- Emphasize the core vision: *"Don't just detect an attack after it happens — forecast the likelihood before it happens."*

### 2. Architecture & Pipeline (Speaker 2 - 1:30 to 3:30)
- Walk through the pipeline: Ingestion $\rightarrow$ Windowed Feature Engineering $\rightarrow$ Behavioral Anomaly Detection $\rightarrow$ Attack Classification $\rightarrow$ Temporal Forecasting.
- Detail why temporal sequences are key: attacks don't happen instantaneously; they follow preparatory stages.

### 3. Live System Demonstration (Speaker 3 - 3:30 to 5:30)
- Showcase the React SOC dashboard.
- Walk through the Reconnaissance $\rightarrow$ Escalation demo scenario.
- Highlight the 5-minute lead time warning provided by the forecast model.

### 4. Technical Novelty & Impact (Speaker 4 - 5:30 to 7:00)
- Highlight advantages over standard NIDS (Snort, Suricata, Zeek).
- Explain encrypted traffic support through pure flow and packet metadata dynamics without DPI.

### 5. Roadmap & Conclusion (Speaker 1 - 7:00 to 8:00)
- Present future extensions: automated SDN firewall rule dispatch and explainable AI insights.
- Open floor for jury questions.
