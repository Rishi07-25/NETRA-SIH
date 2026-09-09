"""Pydantic schemas for security incident events and early-warning alerts."""

from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class SecurityEvent(BaseModel):
    """Structured security alert or forecasted threat event."""

    event_id: str = Field(..., description="Unique event identifier")
    timestamp: datetime = Field(..., description="Timestamp of event occurrence or generation")
    event_type: str = Field(..., description="Type: PRECURSOR_DETECTED, THREAT_ESCALATING, ATTACK_ACTIVE, ANOMALY_FLAG")
    severity: str = Field(..., description="LOW, MEDIUM, HIGH, or CRITICAL")
    title: str = Field(..., description="Short summary title of alert")
    description: str = Field(..., description="Detailed operational alert text")
    details: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Auxiliary metrics or flow attributes")


class EventFeedResponse(BaseModel):
    """Historical and active alert stream."""

    total_events: int = Field(..., description="Total count of events in feed")
    events: List[SecurityEvent] = Field(..., description="List of chronologically ordered security events")
