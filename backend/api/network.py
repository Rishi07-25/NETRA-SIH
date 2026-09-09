"""API route handlers for network telemetry and flow statistics."""

from fastapi import APIRouter
from backend.schemas.network import NetworkStatsResponse
from backend.services.traffic_service import TrafficService

router = APIRouter(prefix="/network", tags=["network"])
traffic_service = TrafficService()


@router.get("/stats", response_model=NetworkStatsResponse)
def get_network_stats():
    """Retrieve real-time network flow statistics and throughput."""
    metrics = traffic_service.get_current_metrics()
    return NetworkStatsResponse(
        active_flows=metrics["active_flows"],
        packets_per_second=metrics["packets_per_second"],
        bytes_per_second=metrics["bytes_per_second"],
        anomaly_rate=metrics["anomaly_rate"],
    )
