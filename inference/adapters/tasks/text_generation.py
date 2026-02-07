from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.rag.prompt_builder import RAGPromptBuilder
from inference.task.processor import TaskProcessor


class TextGenerationTaskAdapter(BaseTaskAdapter):
    """Default adapter for text-in/text-out tasks."""
    name = "text_generation"

    def build_prompts(self, dataset: Dataset, task_cfg: Mapping[str, Any]) -> List[str]:
        """Render prompt templates for each row."""
        rag_cfg = task_cfg.get("rag") or {}
        if not rag_cfg.get("enabled"):
            self._rag_extras = None
            return TaskProcessor.apply_template(
                dataset, task_cfg["prompt_template"], task_cfg["input_mappings"]
            )

        if "{context}" not in task_cfg["prompt_template"]:
            raise ValueError("RAG enabled but prompt_template is missing '{context}'.")

        rag_builder = RAGPromptBuilder(task_cfg)
        prompts, extras = rag_builder.build_prompts(dataset)
        # Store per-example retrieval results for detailed outputs.
        self._rag_extras = extras
        return prompts

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

    def collect_extras(
        self, task_cfg: Mapping[str, Any], count: int
    ) -> List[Mapping[str, Any]] | None:
        """Return saved RAG metadata when available."""
        _ = task_cfg
        extras = getattr(self, "_rag_extras", None)
        if not extras:
            return None
        return list(extras[:count])


class GenericTextTaskAdapter(TextGenerationTaskAdapter):
    """Backward-compatible alias for the text generation adapter."""
    name = "generic"


class ClassificationTaskAdapter(TextGenerationTaskAdapter):
    """Adapter for single-label classification prompts."""
    name = "classification"


class QATaskAdapter(TextGenerationTaskAdapter):
    """Adapter for question-answering prompts."""
    name = "qa"


class SummarizationTaskAdapter(TextGenerationTaskAdapter):
    """Adapter for summarization prompts."""
    name = "summarization"
