from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.model.base_model import BaseModel


class BaseTaskAdapter(ABC):
    """Base interface for task adapters."""
    name = "base"

    @abstractmethod
    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Construct prompts for each dataset row."""
        raise NotImplementedError

    @abstractmethod
    def extract_references(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[Any]:
        """Extract reference targets for metric computation."""
        raise NotImplementedError

    def postprocess_predictions(
        self, predictions: Sequence[str], task_cfg: Mapping[str, Any]
    ) -> List[Any]:
        """Apply task-specific postprocessing to predictions."""
        return list(predictions)

    async def generate_predictions(
        self,
        model: BaseModel,
        prompts: Sequence[str],
        task_cfg: Mapping[str, Any],
        batch_size: int,
        **kwargs,
    ) -> Tuple[List[Any], List[Mapping[str, Any]] | None]:
        """Generate predictions for prompts and optionally return extra metadata."""
        predictions = await model.batch_generate(prompts, batch_size=batch_size, **kwargs)
        return list(predictions), None

    def collect_extras(
        self, task_cfg: Mapping[str, Any], count: int
    ) -> List[Mapping[str, Any]] | None:
        """Return saved per-example metadata if the adapter generates any."""
        _ = (task_cfg, count)
        return None

    def normalize_for_metrics(
        self,
        predictions: Sequence[Any],
        references: Sequence[Any],
        task_cfg: Mapping[str, Any],
    ) -> Tuple[List[Any], List[Any]]:
        """Normalize predictions/references into the format metrics expect."""
        return list(predictions), list(references)
