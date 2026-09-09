"""API route handlers for attack classification and threat prediction."""

from fastapi import APIRouter, HTTPException
from backend.schemas.prediction import PredictionRequest, PredictionResponse
from backend.services.prediction_service import PredictionService

router = APIRouter(prefix="/predict", tags=["prediction"])
prediction_service = PredictionService()


@router.post("/", response_model=PredictionResponse)
def predict_threat(payload: PredictionRequest):
    """Predict attack category for incoming flow feature vectors."""
    try:
        result = prediction_service.classify_threat(payload.features)
        return PredictionResponse(
            attack_class=result["attack_class"],
            confidence=result["confidence"],
            top_features=result["top_features"],
        )
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=f"ML Model unavailable: {str(fnf)}")
    except (ValueError, KeyError, TypeError) as val_err:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(val_err)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal inference error occurred.")
