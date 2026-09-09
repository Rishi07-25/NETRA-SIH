"""Pydantic schemas for behavioral anomaly detection requests and responses."""

from typing import Dict, Optional
from pydantic import BaseModel, Field


class AnomalyRequest(BaseModel):
    """Input features representing a network flow slice for anomaly evaluation."""

    flow_id: Optional[str] = Field(None, description="Unique flow identifier")
    features: Dict[str, float] = Field(..., description="Flow feature dictionary")


class AnomalyResponse(BaseModel):
    """Behavioral anomaly evaluation output."""

    is_anomalous: bool = Field(..., description="True if marked anomalous by Isolation Forest")
    anomaly_score: float = Field(..., ge=0.0, le=1.0, description="Normalized display anomaly score in [0, 1]")
    raw_score: float = Field(..., description="Raw decision score from Isolation Forest")
    model_name: str = Field("IsolationForestAnomalyDetector", description="Name of anomaly detector model")
