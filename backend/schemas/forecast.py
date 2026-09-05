"""Pydantic schemas for attack risk forecasting requests and responses.

Placeholder for future implementation.
"""

from typing import Dict, List
from pydantic import BaseModel, Field


class ForecastRequest(BaseModel):
    """Sequence of historical windowed feature vectors."""

    window_count: int = Field(..., description="Number of past time windows provided")
    history: List[Dict[str, float]] = Field(..., description="Chronological feature windows")


class ForecastHorizon(BaseModel):
    """Forecasted metrics for a specific forward time horizon."""

    horizon: str = Field(..., description="Time horizon, e.g., '1m', '5m', '15m'")
    risk_score: float = Field(..., ge=0.0, le=100.0, description="Risk score (0-100)")
    attack_probability: float = Field(..., ge=0.0, le=1.0, description="Probability of attack")


class ForecastResponse(BaseModel):
    """Risk forecast projections across horizons."""

    current_threat_score: float = Field(..., description="Current unified threat score")
    forecasts: List[ForecastHorizon] = Field(..., description="Risk projections by horizon")
    trajectory: str = Field(..., description="'STABLE', 'ESCALATING', or 'DECLINING'")
