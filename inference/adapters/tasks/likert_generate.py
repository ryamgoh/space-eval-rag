from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.task.processor import TaskProcessor


class LikertGenerateTaskAdapter(BaseTaskAdapter):
    """Likert task adapter that parses numeric ratings from text."""
    name = "likert"

    def __init__(self) -> None:
        """Initialize storage for per-example parsing metadata."""
        self._extras: List[Dict[str, Any]] | None = None

    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Render prompts with the provided template and mappings."""
        return TaskProcessor.apply_template(
            dataset, task_cfg["prompt_template"], task_cfg["input_mappings"]
        )

    def extract_references(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[Any]:
        """Extract reference values for each example."""
        return TaskProcessor.extract_references(dataset, task_cfg)

    def postprocess_predictions(
        self, predictions: Sequence[str], task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Parse Likert ratings from model outputs."""
        predictions = TaskProcessor.postprocess_predictions(list(predictions), task_cfg)
        scale = task_cfg.get("scale") or []
        scale_set = {str(item) for item in scale}
        parsed: List[Any] = []
        extras: List[Dict[str, Any]] = []
        for pred in predictions:
            pred_text = str(pred).strip()
            # Pull the first integer token if it matches the scale.
            match = re.search(r"-?\d+", pred_text)
            value = None
            if match:
                candidate = match.group(0)
                if candidate in scale_set:
                    value = int(candidate) if candidate.isdigit() else candidate
            parsed.append(value if value is not None else pred_text)
            extras.append({"parsed_value": value, "raw_text": pred_text})
        self._extras = extras
        return parsed

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Return predictions/references as-is for scoring."""
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
