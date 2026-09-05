# SIH Problem Statement Analysis

## Problem Statement ID
**SIH26153**

## Title
AI based Network Attack Forecasting from Network Traffic Data

## Organization
**National Technical Research Organisation (NTRO)**

## Theme & Category
- **Theme:** Blockchain & Cybersecurity
- **Category:** Software

---

## 1. Context & Motivation
National security networks, critical infrastructure, and enterprise data backbones face increasingly sophisticated multi-stage cyber threats. Conventional network defense appliances—including Snort/Suricata Network Intrusion Detection Systems (NIDS) and SIEM platforms—predominantly operate **reactively**. They alert security personnel only after signature thresholds are crossed, payloads are delivered, or anomalous payloads are already within internal perimeters.

## 2. The Core Problem
1. **Late Detection Window:** Attacks are identified during or post-breach, resulting in minimal mean time to respond (MTTR).
2. **Alert Fatigue:** SOC analysts are overwhelmed by millions of raw alerts daily, often without context on whether early signs represent an isolated probe or a staged precursor to a targeted attack.
3. **Temporal Blindspots:** Flow metrics are typically treated independently rather than as sequential temporal dynamics evolving along the cyber attack kill chain.

## 3. Objective of NETRA
Develop an AI-driven system capable of processing network traffic data (packet traces, flow telemetry) over temporal sliding windows to forecast the likelihood, timing, and category of imminent cyber attacks before catastrophic impact occurs.

## 4. Expected Deliverables
- Ingestion and parsing engine for live or sampled network flow telemetry.
- Machine learning models for early-stage anomaly detection, signature-free classification, and time-series risk forecasting.
- Unified risk and threat scoring model for SOC prioritization.
- Mission-ready analytical user interface visualizing network health, risk trajectory, and early-warning alerts.
