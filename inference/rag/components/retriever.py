from __future__ import annotations

from typing import Any, Dict, List

from inference.rag.manager import RAGManager


class Retriever:
    """Retrieve context from an initialized RAG manager."""
    def __init__(self, manager: RAGManager):
        self._manager = manager

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Return retrieved documents for a query."""
        return self._manager.retrieve(query, k)
