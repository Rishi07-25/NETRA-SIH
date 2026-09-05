"""API route handlers for attack classification and threat prediction.

Placeholder for future implementation.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/predict", tags=["prediction"])


@router.post("/")
def predict_threat():
    """Predict attack category for incoming flow feature vectors."""
    raise NotImplementedError("Threat prediction endpoint will be implemented in future work.")
