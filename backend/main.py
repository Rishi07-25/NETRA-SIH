"""NETRA FastAPI Application Entrypoint.

Provides RESTful endpoints for attack classification, behavioral anomaly detection,
temporal risk forecasting, real-time traffic statistics, and security event alerts.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.anomaly import router as anomaly_router
from backend.api.events import router as events_router
from backend.api.forecast import router as forecast_router
from backend.api.network import router as network_router
from backend.api.predict import router as predict_router

app = FastAPI(
    title="NETRA - Network Threat Early-warning & Risk Analytics",
    description="Backend API for AI-based network attack forecasting and risk telemetry.",
    version="0.2.0",
)

# CORS Middleware setup for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers with unified prefix support
app.include_router(predict_router)
app.include_router(forecast_router)
app.include_router(anomaly_router)
app.include_router(network_router)
app.include_router(events_router)

# Also support /api/v1 prefix as documented in docs/api-documentation.md
app.include_router(predict_router, prefix="/api/v1")
app.include_router(forecast_router, prefix="/api/v1")
app.include_router(anomaly_router, prefix="/api/v1")
app.include_router(network_router, prefix="/api/v1")
app.include_router(events_router, prefix="/api/v1")


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "NETRA Backend API",
        "version": "0.2.0",
        "mode": "production_ready",
        "capabilities": [
            "attack_classification",
            "behavioral_anomaly_detection",
            "temporal_risk_forecasting",
            "network_telemetry",
            "security_event_feed",
        ],
    }
