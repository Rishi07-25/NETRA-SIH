"""API route handlers for behavioral anomaly detection.

Placeholder for future implementation.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/anomaly", tags=["anomaly"])


@router.post("/")
def detect_anomaly():
    """Evaluate network traffic slice for behavioral anomalies."""
    raise NotImplementedError("Anomaly detection endpoint will be implemented in future work.")
