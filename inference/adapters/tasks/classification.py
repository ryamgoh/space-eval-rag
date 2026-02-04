from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.task.processor import TaskProcessor


class ClassificationTaskAdapter(BaseTaskAdapter):
    """Adapter for classification tasks with label normalization."""
    name = "classification"

    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Render classification prompts for each row."""
        return TaskProcessor.apply_template(
            dataset, task_cfg["prompt_template"], task_cfg["input_mappings"]
        )

    def extract_references(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[Any]:
        """Extract reference labels from dataset rows."""
        return TaskProcessor.extract_references(dataset, task_cfg)

    def postprocess_predictions(
        self, predictions: Sequence[str], task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Apply configured post-processing to model outputs."""
        return TaskProcessor.postprocess_predictions(list(predictions), task_cfg)

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Normalize predictions/references to numeric label IDs."""
        label_map = task_cfg.get("label_map")
        if not label_map:
            raise ValueError("Classification task adapter requires label_map.")
        return TaskProcessor.normalize_classification(
            list(predictions), list(references), label_map
        )
