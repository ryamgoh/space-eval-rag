from __future__ import annotations

from typing import Any, Dict, List, Optional

from inference.config.models import RAGConfig
from inference.rag.embeddings import HFEmbeddingModel
from inference.rag.index import FAISSIndex


class RAGManager:
    """Retrieval-augmented generation manager using FAISS."""

    def __init__(self, config: RAGConfig):
        """Initialize RAG configuration without building an index yet."""
        self.config = config
        self._embedder: Optional[HFEmbeddingModel] = None
        self._index: Optional[FAISSIndex] = None

    def _get_embedder(self) -> HFEmbeddingModel:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            if self.config.embedding_model:
                self._embedder = HFEmbeddingModel(self.config.embedding_model)
        return self._embedder

    def build_index(
        self,
        documents: List[str],
    ) -> None:
        """Build a FAISS index for the provided documents."""
        embedder = self._get_embedder()
        self._index = FAISSIndex.build(
            documents=list(documents),
            embed_fn=embedder.embed_texts,
        )

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query."""
        if not query.strip():
            return []
        if self._index is None:
            raise RuntimeError("RAG index is not initialized.")
        embedder = self._get_embedder()
        query_vec = embedder.embed_texts([query])
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(self._index.search(query_vec, k), start=1):
            results.append(
                {
                    "id": int(idx),
                    "rank": rank,
                    "score": float(score),
                    "text": self._index.docs[idx],
                }
            )
        return results
