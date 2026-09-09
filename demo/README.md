# Demonstration Playbook & Assets

## Overview
This directory packages demonstration scripts, end-to-end rehearsal workflows, sample evaluation runs, and presentation captures for hackathon presentations and evaluator reviews.

## Components & Scripts
- `run_demo.py`: Automated, executable end-to-end demo runner validating the entire progression from baseline network health through reconnaissance to credential brute-force escalation.
- `demo-scenario.md`: Structured walkthrough script executing the end-to-end demonstration.
- `demo-data/`: Curated sample data slices designed to trigger deterministic forecasting progressions during live demos.
- `screenshots/`: UI captures of the SOC dashboard, threat timeline, and forecast cards.
- `recordings/`: Screen recordings and video walkthroughs of the operational system.

## Running the Demonstration
```bash
# Execute end-to-end demo pipeline
python -m demo.run_demo
```
