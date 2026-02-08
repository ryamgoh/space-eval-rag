from __future__ import annotations

from typing import Any, List, Mapping

from inference.task.processor import TaskProcessor


class CorpusLoader:
    """Load and render corpus documents for RAG indexing."""
    def __init__(self, rag_cfg: Mapping[str, Any]):
        self._rag_cfg = rag_cfg

    def load(self) -> List[str]:
        """Render each corpus row into a string to index."""
        corpus_cfg = self._rag_cfg["corpus"]
        corpus_split = self._rag_cfg.get("corpus_split")
        corpus_dataset = TaskProcessor.load_dataset(corpus_cfg, split=corpus_split)
        return TaskProcessor.apply_template(
            corpus_dataset, self._rag_cfg["corpus_template"], self._rag_cfg["corpus_mappings"]
        )
