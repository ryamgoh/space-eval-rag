from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping

from datasets import Dataset

from inference.rag.manager import RAGManager
from inference.task.processor import TaskProcessor


class RAGPromptBuilder:
    """Build prompts with retrieved context for a single task."""
    def __init__(self, task_cfg: Mapping[str, Any]):
        """Store task/RAG configuration."""
        self.task_cfg = task_cfg
        self.rag_cfg = task_cfg.get("rag") or {}
        if not self.rag_cfg.get("enabled"):
            raise ValueError("RAGPromptBuilder requires rag.enabled: true.")

    def build_prompts(self, dataset: Dataset) -> tuple[List[str], List[Mapping[str, Any]]]:
        """Build prompts and per-example retrieval metadata."""
        corpus_docs = self._build_corpus_docs()
        rag_manager = self._build_manager(corpus_docs)
        return self._build_prompts(dataset, rag_manager)

    def _build_corpus_docs(self) -> List[str]:
        """Render each corpus row into a string to index."""
        corpus_cfg = self.rag_cfg["corpus"]
        corpus_split = self.rag_cfg.get("corpus_split")
        corpus_dataset = TaskProcessor.load_dataset(corpus_cfg, split=corpus_split)
        return TaskProcessor.apply_template(
            corpus_dataset, self.rag_cfg["corpus_template"], self.rag_cfg["corpus_mappings"]
        )

    def _build_manager(self, corpus_docs: List[str]) -> RAGManager:
        """Create a manager and build/load the FAISS index."""
        rag_manager = RAGManager(self.rag_cfg)
        rag_manager.build_or_load_index(
            corpus_docs,
            cache_dir=self.rag_cfg.get("cache_dir"),
            cache_key=self.rag_cfg.get("cache_key"),
        )
        return rag_manager

    def _build_prompts(
        self,
        dataset: Iterable[Mapping[str, Any]],
        rag_manager: RAGManager,
    ) -> tuple[List[str], List[Mapping[str, Any]]]:
        """Retrieve context per row and render the final prompts."""
        query_template = self.rag_cfg["query_template"]
        query_mappings = self.rag_cfg["query_mappings"]
        context_k = self.rag_cfg["context_k"]
        context_template = self.rag_cfg["context_template"]
        context_separator = self.rag_cfg["context_separator"]

        prompts: List[str] = []
        extras: List[Dict[str, Any]] = []
        for row in dataset:
            query = TaskProcessor.render_template(row, query_template, query_mappings)
            retrieved = rag_manager.retrieve(query, context_k)
            context = rag_manager.format_context(retrieved, context_template, context_separator)
            prompt = TaskProcessor.render_template(
                row,
                self.task_cfg["prompt_template"],
                self.task_cfg["input_mappings"],
                extras={"context": context},
            )
            prompts.append(prompt)
            extras.append(
                {
                    "rag": {
                        "query": query,
                        "context": context,
                        "results": retrieved,
                    }
                }
            )
        return prompts, extras
