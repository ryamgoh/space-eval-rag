from __future__ import annotations

from typing import Any, Dict, Sequence

import evaluate

from inference.adapters.base_metric import BaseMetricAdapter


class HFEvaluateMetricAdapter(BaseMetricAdapter):
    """Metric adapter that delegates to Hugging Face evaluate."""

    name = "hf"

    def compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        metric_name: str,
        metric_args: dict[str, Any] | None = None,
    ) -> Dict[str, float]:
        """Compute metrics using evaluate.load(...)."""
        metric_args = metric_args or {}
        metric = evaluate.load(metric_name)
        result = metric.compute(
            predictions=list(predictions), references=list(references), **metric_args
        )
        return {f"{metric_name}:{key}": value for key, value in result.items()}
