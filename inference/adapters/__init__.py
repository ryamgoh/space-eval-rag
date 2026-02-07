from inference.adapters.registry import metric_adapters, task_adapters
from inference.adapters.metrics.hf import HFEvaluateMetricAdapter
from inference.adapters.tasks.text_generation import (
    ClassificationTaskAdapter,
    GenericTextTaskAdapter,
    QATaskAdapter,
    SummarizationTaskAdapter,
    TextGenerationTaskAdapter,
)

task_adapters.register(TextGenerationTaskAdapter.name, TextGenerationTaskAdapter)
task_adapters.register(GenericTextTaskAdapter.name, GenericTextTaskAdapter)
task_adapters.register(ClassificationTaskAdapter.name, ClassificationTaskAdapter)
task_adapters.register(QATaskAdapter.name, QATaskAdapter)
task_adapters.register(SummarizationTaskAdapter.name, SummarizationTaskAdapter)

metric_adapters.register(HFEvaluateMetricAdapter.name, HFEvaluateMetricAdapter)

__all__ = [
    "metric_adapters",
    "task_adapters",
]
