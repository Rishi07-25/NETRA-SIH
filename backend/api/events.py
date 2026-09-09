"""API route handlers for security events and early-warning alerts."""

from typing import List
from fastapi import APIRouter
from backend.schemas.events import EventFeedResponse, SecurityEvent
from backend.services.traffic_service import TrafficService

router = APIRouter(prefix="/events", tags=["events"])
traffic_service = TrafficService()


@router.get("/", response_model=EventFeedResponse)
def list_security_events():
    """Retrieve historical and live threat events feed."""
    raw_events = traffic_service.get_events()
    events = [
        SecurityEvent(
            event_id=e["event_id"],
            timestamp=e["timestamp"],
            event_type=e["event_type"],
            severity=e["severity"],
            title=e["title"],
            description=e["description"],
            details=e.get("details", {}),
        )
        for e in raw_events
    ]
    return EventFeedResponse(
        total_events=len(events),
        events=events,
    )
