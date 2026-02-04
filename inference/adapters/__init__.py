from inference.adapters.registry import metric_adapters, task_adapters
from inference.adapters.metrics.hf import HFEvaluateMetricAdapter
from inference.adapters.tasks.generic import GenericTextTaskAdapter

task_adapters.register(GenericTextTaskAdapter.name, GenericTextTaskAdapter)

metric_adapters.register(HFEvaluateMetricAdapter.name, HFEvaluateMetricAdapter)

__all__ = [
    "metric_adapters",
    "task_adapters",
]
