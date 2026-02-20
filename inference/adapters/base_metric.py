from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence

from inference.config.models import MetricConfig


class BaseMetricAdapter(ABC):
    """Base interface for metric adapters."""

    name = "base"

    @abstractmethod
    def compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        metric_name: str,
        metric_args: dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        """Compute metric values for predictions and references."""
        raise NotImplementedError
