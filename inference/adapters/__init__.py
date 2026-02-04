from inference.adapters.registry import metric_adapters, task_adapters
from inference.adapters.metrics.hf import HFEvaluateMetricAdapter
from inference.adapters.tasks.choice_generate import ChoiceGenerateTaskAdapter
from inference.adapters.tasks.choice_logprob import ChoiceLogProbTaskAdapter
from inference.adapters.tasks.classification import ClassificationTaskAdapter
from inference.adapters.tasks.generic import GenericTextTaskAdapter
from inference.adapters.tasks.likert_generate import LikertGenerateTaskAdapter
from inference.adapters.tasks.likert_logprob import LikertLogProbTaskAdapter

task_adapters.register(GenericTextTaskAdapter.name, GenericTextTaskAdapter)
task_adapters.register(ClassificationTaskAdapter.name, ClassificationTaskAdapter)
task_adapters.register(ChoiceGenerateTaskAdapter.name, ChoiceGenerateTaskAdapter)
task_adapters.register(ChoiceLogProbTaskAdapter.name, ChoiceLogProbTaskAdapter)
task_adapters.register(LikertGenerateTaskAdapter.name, LikertGenerateTaskAdapter)
task_adapters.register(LikertLogProbTaskAdapter.name, LikertLogProbTaskAdapter)

metric_adapters.register(HFEvaluateMetricAdapter.name, HFEvaluateMetricAdapter)

__all__ = [
    "metric_adapters",
    "task_adapters",
]
