from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping, Sequence


class BaseMetricAdapter(ABC):
    """Base interface for metric adapters."""
    name = "base"

    @abstractmethod
    def compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        metric_cfg: Mapping[str, Any] | str,
    ) -> Dict[str, float]:
        """Compute metric values for predictions and references."""
        raise NotImplementedError
