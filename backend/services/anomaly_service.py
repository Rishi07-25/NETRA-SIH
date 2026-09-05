"""Service managing behavioral anomaly detection.

Placeholder for future implementation.
"""


class AnomalyService:
    """Service orchestrator for behavioral anomaly scoring."""

    def __init__(self, model_path: str = None):
        self.model_path = model_path

    def evaluate_anomaly(self, flow_sample: dict):
        """Calculate anomaly score for the given network flow sample."""
        raise NotImplementedError("AnomalyService logic will be implemented in future work.")
