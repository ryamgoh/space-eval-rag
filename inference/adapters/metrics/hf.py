from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

import evaluate

from inference.adapters.base_metric import BaseMetricAdapter


class HFEvaluateMetricAdapter(BaseMetricAdapter):
    """Metric adapter that delegates to Hugging Face evaluate."""
    name = "hf"

    def compute(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        metric_cfg: Mapping[str, Any] | str,
    ) -> Dict[str, float]:
        """Compute metrics using evaluate.load(...)."""
        if isinstance(metric_cfg, str):
            name = metric_cfg
            metric_args: Dict[str, Any] = {}
        else:
            name = metric_cfg["name"]
            metric_args = metric_cfg.get("args", {})

        metric = evaluate.load(name)
        result = metric.compute(
            predictions=list(predictions), references=list(references), **metric_args
        )
        return {f"{name}:{key}": value for key, value in result.items()}
