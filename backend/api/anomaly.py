"""API route handlers for behavioral anomaly detection."""

from fastapi import APIRouter, HTTPException
from backend.schemas.anomaly import AnomalyRequest, AnomalyResponse
from backend.services.anomaly_service import AnomalyService

router = APIRouter(prefix="/anomaly", tags=["anomaly"])
anomaly_service = AnomalyService()


@router.post("/", response_model=AnomalyResponse)
def detect_anomaly(payload: AnomalyRequest):
    """Evaluate network traffic slice for behavioral anomalies."""
    try:
        res = anomaly_service.evaluate_anomaly(payload.features)
        return AnomalyResponse(
            is_anomalous=res["is_anomalous"],
            anomaly_score=res["anomaly_score"],
            raw_score=res["raw_score"],
            model_name=res["model_name"],
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=f"Anomaly Model unavailable: {str(fnf)}")
    except (ValueError, KeyError, TypeError) as val_err:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(val_err)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal anomaly detection error occurred.")
