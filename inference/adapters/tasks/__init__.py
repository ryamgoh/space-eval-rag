from inference.adapters.tasks.choice_generate import ChoiceGenerateTaskAdapter
from inference.adapters.tasks.choice_logprob import ChoiceLogProbTaskAdapter
from inference.adapters.tasks.classification import ClassificationTaskAdapter
from inference.adapters.tasks.generic import GenericTextTaskAdapter
from inference.adapters.tasks.likert_generate import LikertGenerateTaskAdapter
from inference.adapters.tasks.likert_logprob import LikertLogProbTaskAdapter

__all__ = [
    "ChoiceGenerateTaskAdapter",
    "ChoiceLogProbTaskAdapter",
    "ClassificationTaskAdapter",
    "GenericTextTaskAdapter",
    "LikertGenerateTaskAdapter",
    "LikertLogProbTaskAdapter",
]
