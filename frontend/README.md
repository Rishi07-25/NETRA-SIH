# Frontend SOC Dashboard

## Overview
The `frontend/` directory contains the React application providing an interactive Security Operations Center (SOC) dashboard. It displays live network metrics, early-warning risk projections, threat score timelines, and active incident tables communicating with the NETRA FastAPI backend.

## Architecture & Components
- `src/components/ThreatScore.jsx`: Visual threat severity gauge and status badges.
- `src/components/ForecastCard.jsx`: Multi-horizon forward risk probability and attack stage cards.
- `src/components/NetworkStats.jsx`: Real-time packet throughput, bandwidth, and anomaly rate cards.
- `src/components/ThreatTimeline.jsx`: Chronological attack trajectory bar chart and trend indicator.
- `src/components/AttackTable.jsx`: Active incident classifications and attribution table.
- `src/components/EventFeed.jsx`: Early-warning security alert stream.
- `src/pages/Dashboard.jsx`: Central operational dashboard container.
- `src/services/api.js`: Asynchronous REST client for FastAPI endpoints.

## Local Development & Build
```bash
cd frontend

# Install dependencies
npm install

# Start Vite development server
npm run dev

# Build production bundle
npm run build
```
