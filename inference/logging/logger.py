from __future__ import annotations

from datetime import datetime
from typing import Dict, List


class Monitor:
    def log_metrics(self, model_name: str, metrics: Dict, timestamp: datetime) -> None:
        _ = (model_name, metrics, timestamp)

    def track_resources(self) -> Dict:
        return {}

    def create_dashboard(self, metrics_history: List[Dict]) -> None:
        _ = metrics_history
