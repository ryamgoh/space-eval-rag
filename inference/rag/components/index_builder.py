from __future__ import annotations

from inference.config.models import RAGConfig
from inference.rag.components.corpus_loader import CorpusLoader
from inference.rag.manager import RAGManager


class IndexBuilder:
    """Build or load a RAG manager for a corpus."""

    def __init__(self, rag_cfg: RAGConfig, corpus_loader: CorpusLoader):
        self._rag_cfg = rag_cfg
        self._corpus_loader = corpus_loader

    def build(self) -> RAGManager:
        """Build or load a manager for the configured corpus."""
        manager = RAGManager(self._rag_cfg)
        manager.build_index(self._corpus_loader.load())
        return manager
