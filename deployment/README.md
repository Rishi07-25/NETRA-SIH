# Deployment & Containerization

## Overview
This directory contains infrastructure configurations, container Dockerfiles, and reverse-proxy definitions for local development and production deployments.

## Directory Components
- `Dockerfile.backend`: Multi-stage Python build containerizing the FastAPI service.
- `Dockerfile.frontend`: Multi-stage Node/Nginx container building and serving the React SOC dashboard.
- `nginx.conf`: Nginx reverse proxy configuration routing frontend assets and API requests.

## Deployment Instructions
To build and deploy the complete stack using Docker Compose:
```bash
docker compose -f docker-compose.yml up --build -d
```

## Primary Owner
**Workstream 4: Backend/API** (In collaboration with Workstream 5)
