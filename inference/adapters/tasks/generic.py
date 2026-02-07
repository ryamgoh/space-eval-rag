from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from datasets import Dataset

from inference.adapters.base_task import BaseTaskAdapter
from inference.rag.manager import RAGManager
from inference.task.processor import TaskProcessor


class GenericTextTaskAdapter(BaseTaskAdapter):
    """Default adapter for text-in/text-out tasks."""
    name = "generic"

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

        corpus_cfg = rag_cfg["corpus"]
        corpus_split = rag_cfg.get("corpus_split")
        corpus_dataset = TaskProcessor.load_dataset(corpus_cfg, split=corpus_split)
        # Render each corpus row into a string that will be indexed.
        corpus_docs = TaskProcessor.apply_template(
            corpus_dataset, rag_cfg["corpus_template"], rag_cfg["corpus_mappings"]
        )

        rag_manager = RAGManager(rag_cfg)
        rag_manager.build_or_load_index(
            corpus_docs,
            cache_dir=rag_cfg.get("cache_dir"),
            cache_key=rag_cfg.get("cache_key"),
        )

        query_template = rag_cfg.get("query_template", task_cfg["prompt_template"].replace("{context}", ""))
        query_mappings = rag_cfg.get("query_mappings", task_cfg["input_mappings"])
        context_k = rag_cfg.get("context_k", 3)
        context_template = rag_cfg.get("context_template", "{text}")
        context_separator = rag_cfg.get("context_separator", "\n\n")

        prompts: List[str] = []
        # Store per-example retrieval results for detailed outputs.
        self._rag_extras = []
        for row in dataset:
            query = TaskProcessor.render_template(row, query_template, query_mappings)
            retrieved = rag_manager.retrieve(query, context_k)
            context = rag_manager.format_context(retrieved, context_template, context_separator)
            prompt = TaskProcessor.render_template(
                row,
                task_cfg["prompt_template"],
                task_cfg["input_mappings"],
                extras={"context": context},
            )
            prompts.append(prompt)
            self._rag_extras.append(
                {
                    "rag": {
                        "query": query,
                        "context": context,
                        "results": retrieved,
                    }
                }
            )
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
