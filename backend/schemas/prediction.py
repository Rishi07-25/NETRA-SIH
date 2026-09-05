"""Pydantic schemas for threat classification requests and responses.

Placeholder for future implementation.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    """Input features representing network flow attributes."""

    flow_id: Optional[str] = Field(None, description="Unique flow identifier")
    features: Dict[str, float] = Field(..., description="Flow feature dictionary")


class PredictionResponse(BaseModel):
    """Output prediction containing attack category and confidence."""

    attack_class: str = Field(..., description="Predicted attack category or 'Benign'")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score")
    top_features: Optional[List[str]] = Field(default=[], description="Top contributing features")
