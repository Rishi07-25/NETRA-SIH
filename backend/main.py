"""NETRA FastAPI Application Entrypoint.

Placeholder application structure for future implementation.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="NETRA - Network Threat Early-warning & Risk Analytics",
    description="Backend API for AI-based network attack forecasting and risk telemetry.",
    version="0.1.0",
)

# CORS Middleware setup for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def health_check():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "NETRA Backend API",
        "version": "0.1.0",
        "mode": "initial_setup",
    }
