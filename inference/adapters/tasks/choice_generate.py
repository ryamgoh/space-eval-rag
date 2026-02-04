from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.task.processor import TaskProcessor


class ChoiceGenerateTaskAdapter(BaseTaskAdapter):
    """Choice task adapter that parses labels from generated text."""
    name = "choice"

    def __init__(self) -> None:
        """Initialize storage for per-example parsing metadata."""
        self._extras: List[Dict[str, Any]] | None = None

    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Render prompts with the provided template and mappings."""
        return TaskProcessor.apply_template(
            dataset, task_cfg["prompt_template"], task_cfg["input_mappings"]
        )

    def extract_references(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[Any]:
        """Extract reference labels for each example."""
        return TaskProcessor.extract_references(dataset, task_cfg)

    def postprocess_predictions(
        self, predictions: Sequence[str], task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Parse a choice label from the model's generated text."""
        predictions = TaskProcessor.postprocess_predictions(list(predictions), task_cfg)
        choices = task_cfg.get("choices") or []
        parsed: List[Any] = []
        extras: List[Dict[str, Any]] = []
        for pred in predictions:
            pred_text = str(pred).strip()
            matched = None
            for choice in choices:
                # Match by substring so "Label: Sports" still parses correctly.
                if choice.lower() in pred_text.lower():
                    matched = choice
                    break
            parsed.append(matched if matched is not None else pred_text)
            extras.append({"parsed_choice": matched, "raw_text": pred_text})
        self._extras = extras
        return parsed

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Normalize labels to numeric IDs if a label_map is provided."""
        label_map = task_cfg.get("label_map")
        if label_map:
            return TaskProcessor.normalize_classification(
                list(predictions), list(references), label_map
            )
        return list(predictions), list(references)

    async def generate_predictions(
        self,
        model,
        prompts: Sequence[str],
        task_cfg: Mapping[str, Any],
        batch_size: int,
        **kwargs,
    ) -> Tuple[List[Any], List[Mapping[str, Any]] | None]:
        """Generate raw text predictions using the model backend."""
        predictions = await model.batch_generate(prompts, batch_size=batch_size, **kwargs)
        return list(predictions), None

    def collect_extras(
        self, task_cfg: Mapping[str, Any], count: int
    ) -> List[Mapping[str, Any]] | None:
        """Return stored parsing metadata for detailed outputs."""
        if not self._extras:
            return None
        return self._extras[:count]
