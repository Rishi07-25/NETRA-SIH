"""API route handlers for temporal attack risk forecasting.

Placeholder for future implementation.
"""

from fastapi import APIRouter

router = APIRouter(prefix="/forecast", tags=["forecasting"])


@router.post("/")
def forecast_risk():
    """Forecast future attack risk trajectory across time horizons."""
    raise NotImplementedError("Risk forecasting endpoint will be implemented in future work.")
