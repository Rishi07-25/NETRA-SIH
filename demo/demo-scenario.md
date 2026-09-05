# Demonstration Scenario: Reconnaissance to Escalation

## Objective
Demonstrate NETRA's capability to detect low-level reconnaissance precursors and forecast an escalating attack risk before a major incident matures.

---

## Scripted Demo Flow

### Step 1: Baseline Network Health ($T = 0\text{m}$)
- **Action:** Stream benign baseline traffic (`normal.json`).
- **SOC View:** Composite Threat Score sits in the green zone ($< 15$). All horizon forecast cards indicate stable conditions ($P(\text{Attack}) < 5\%$).

### Step 2: Precursor Reconnaissance Injected ($T = 2\text{m}$)
- **Action:** Inject stealth horizontal port scanning (`reconnaissance.json`).
- **SOC View:** Anomaly engine flags anomalous SYN ratios. Attack classification attributes activity to `Reconnaissance` (Confidence: 85%). Threat Score rises moderately ($25 - 35$).

### Step 3: Risk Horizon Forecasting ($T = 4\text{m}$)
- **Action:** Network generator introduces preliminary authentication probes (`escalation.json`).
- **SOC View:** 
  - The **5-minute horizon forecast card** surges to $68\%$ probability.
  - Trajectory switches to **ESCALATING**.
  - Early-warning banner triggers: *"High probability of imminent credential brute-force attack on Port 22 within 5 minutes."*

### Step 4: Full-Scale Brute Force Attempted ($T = 8\text{m}$)
- **Action:** High-rate dictionary attack launched against port 22.
- **SOC View:** Attack occurs exactly as forecasted. Threat Score hits $92/100$. Analyst demonstrates that early-warning alert preceded the actual volumetric incident by several minutes.
