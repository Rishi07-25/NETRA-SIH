# Frontend SOC Dashboard

## Overview
The `frontend/` directory contains the React application providing an interactive Security Operations Center (SOC) dashboard. It displays live network metrics, early-warning risk projections, threat score timelines, and active incident tables.

## Architecture
- `src/components/`: Modular UI widgets (threat score gauge, forecast cards, timeline, attack table, live event stream).
- `src/pages/`: Page containers (primary `Dashboard.jsx`).
- `src/services/`: HTTP client (`api.js`) communicating with backend REST endpoints.
- `public/`: Static assets and favicon files.

## Primary Owner
**Workstream 5: Frontend/SOC Dashboard**
