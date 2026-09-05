"""Pydantic schemas for network flow metrics and statistics.

Placeholder for future implementation.
"""

from pydantic import BaseModel, Field


class NetworkStatsResponse(BaseModel):
    """Real-time network traffic and telemetry statistics."""

    active_flows: int = Field(..., description="Count of currently active network sessions")
    packets_per_second: float = Field(..., description="Current packet throughput")
    bytes_per_second: float = Field(..., description="Current bandwidth throughput")
    anomaly_rate: float = Field(..., ge=0.0, le=1.0, description="Proportion of flagged anomalies")
