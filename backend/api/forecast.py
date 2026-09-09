"""API route handlers for temporal attack risk forecasting."""

from fastapi import APIRouter, HTTPException
from backend.schemas.forecast import ForecastHorizon, ForecastRequest, ForecastResponse
from backend.services.forecasting_service import ForecastingService

router = APIRouter(prefix="/forecast", tags=["forecasting"])
forecasting_service = ForecastingService()


@router.post("/", response_model=ForecastResponse)
def forecast_risk(payload: ForecastRequest):
    """Forecast future attack risk trajectory across time horizons."""
    try:
        if not payload.history:
            raise HTTPException(status_code=422, detail="History windows must not be empty.")

        res = forecasting_service.generate_risk_forecast(payload.history)
        horizons = [
            ForecastHorizon(
                horizon=f["horizon"],
                risk_score=f["risk_score"],
                attack_probability=f["attack_probability"],
            )
            for f in res["forecasts"]
        ]
        return ForecastResponse(
            current_threat_score=res["current_threat_score"],
            forecasts=horizons,
            trajectory=res["trajectory"],
        )
    except HTTPException:
        raise
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=503, detail=f"Forecasting Model unavailable: {str(fnf)}")
    except (ValueError, KeyError, TypeError) as val_err:
        raise HTTPException(status_code=422, detail=f"Validation error: {str(val_err)}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Internal forecasting error occurred.")
