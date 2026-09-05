# Smart India Hackathon (SIH) Jury Q&A Guide

## Core Questions & Defenses

### Q1: How is "forecasting" different from standard "detection"?
**Answer:**
Standard intrusion detection evaluates traffic either packet-by-packet or flow-by-flow at timestamp $t$ and answers: *"Is this malicious right now?"* 
In contrast, NETRA models sequential trends across rolling time windows ($t-W \dots t$) to answer: *"Given the preparatory activity observed over the past $N$ minutes, what is the probability and category of an escalated attack occurring at $t + \Delta t$?"* This early warning window allows defenders to prepare or automate containment.

---

### Q2: What prevents excessive false alarms in dynamic networks?
**Answer:**
NETRA employs a two-pronged validation mechanism:
1. Unsupervised baseline modeling adapts to regular circadian network cycles.
2. A multi-feature correlation filter requires sustained temporal escalation patterns rather than transient single-packet anomalies before elevating the overall Threat Score.

---

### Q3: How do you handle encrypted traffic (TLS 1.3 / HTTPS)?
**Answer:**
NETRA does not rely on Deep Packet Inspection (DPI) or unencrypted payload contents. All features are derived from **packet dynamics and metadata**: inter-arrival times, packet length distributions, flow duration, byte exchange ratios, and directional burst entropy, which remain fully observable in encrypted traffic.

---

### Q4: Can NETRA operate in high-throughput enterprise networks?
**Answer:**
Yes. The architecture decouples high-speed flow aggregation from ML inference. Network taps generate standard IPFIX/NetFlow records, while sliding window summaries pass to lightweight, optimized inference models designed for sub-second latency.
