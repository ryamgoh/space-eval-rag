from __future__ import annotations

from typing import Callable, List, Tuple

import faiss
import numpy as np


class FAISSIndex:
    """FAISS index plus its backing document list."""
    def __init__(self, index: faiss.Index, docs: List[str]):
        self.index = index
        self.docs = docs

    @classmethod
    def build(
        cls,
        documents: List[str],
        embed_fn: Callable[[List[str]], np.ndarray],
    ) -> "FAISSIndex":
        """Build a FAISS index for the provided documents."""
        if not documents:
            raise ValueError("RAG corpus is empty; cannot build index.")

        embeddings = embed_fn(documents)
        if embeddings.size == 0:
            raise ValueError("Failed to embed RAG corpus.")
        dim = embeddings.shape[1]
        # Inner-product index works with normalized embeddings for cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return cls(index=index, docs=list(documents))

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Return (doc_id, score) pairs for a single query vector."""
        if query_vec.size == 0:
            return []
        k = min(int(k), len(self.docs))
        scores, indices = self.index.search(query_vec, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]

