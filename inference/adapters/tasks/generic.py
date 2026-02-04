from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.task.processor import TaskProcessor


class GenericTextTaskAdapter(BaseTaskAdapter):
    """Default adapter for text-in/text-out tasks."""
    name = "generic"

    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Render prompt templates for each row."""
        return TaskProcessor.apply_template(
            dataset, task_cfg["prompt_template"], task_cfg["input_mappings"]
        )

    def extract_references(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[Any]:
        """Extract reference targets from dataset rows."""
        return TaskProcessor.extract_references(dataset, task_cfg)

    def postprocess_predictions(
        self, predictions: Sequence[str], task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Apply configured prediction post-processing."""
        return TaskProcessor.postprocess_predictions(list(predictions), task_cfg)

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Normalize predictions/references for metric computation."""
        label_map = task_cfg.get("label_map")
        if label_map:
            return TaskProcessor.normalize_classification(
                list(predictions), list(references), label_map
            )
        return list(predictions), list(references)
