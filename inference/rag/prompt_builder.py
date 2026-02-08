from __future__ import annotations

from typing import Dict, List, Mapping

from datasets import Dataset

from inference.rag.components.corpus_loader import CorpusLoader
from inference.rag.components.index_builder import IndexBuilder
from inference.rag.components.prompt_renderer import PromptRenderer
from inference.rag.components.retriever import Retriever
from inference.task.processor import TaskProcessor


class RAGPromptBuilder:
    """Build prompts with retrieved context for a single task."""
    def __init__(self, task_cfg: Mapping[str, Any]):
        """Store task/RAG configuration."""
        self.task_cfg = task_cfg
        self.rag_cfg = task_cfg.get("rag") or {}
        if not self.rag_cfg.get("enabled"):
            raise ValueError("RAGPromptBuilder requires rag.enabled: true.")
        self._corpus_loader = CorpusLoader(self.rag_cfg)
        self._index_builder = IndexBuilder(self.rag_cfg, self._corpus_loader)
        self._prompt_renderer = PromptRenderer(self.task_cfg, self.rag_cfg)

    def build_prompts(self, dataset: Dataset) -> tuple[List[str], List[Mapping[str, Any]]]:
        """Build prompts and per-example retrieval metadata."""
        rag_manager = self._index_builder.build()
        retriever = Retriever(rag_manager)

        query_template = self.rag_cfg["query_template"]
        query_mappings = self.rag_cfg["query_mappings"]
        context_k = self.rag_cfg["context_k"]
        context_template = self.rag_cfg["context_template"]
        context_separator = self.rag_cfg["context_separator"]

        prompts: List[str] = []
        extras: List[Dict[str, Any]] = []
        for row in dataset:
            query = TaskProcessor.render_template(row, query_template, query_mappings)
            retrieved = retriever.retrieve(query, context_k)
            context = self._prompt_renderer.format_context(
                retrieved,
                context_template,
                context_separator,
            )
            prompt, extra = self._prompt_renderer.render_prompt(
                row,
                context,
                query,
                retrieved,
            )
            prompts.append(prompt)
            extras.append(extra)
        return prompts, extras
