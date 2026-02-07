from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, List, Mapping, Optional

from inference.rag.embeddings import HFEmbeddingModel
from inference.rag.index import FAISSIndex


class RAGManager:
    """Retrieval-augmented generation manager using FAISS."""
    def __init__(self, config: Mapping[str, Any]):
        """Initialize RAG configuration without building an index yet."""
        self.config = dict(config)
        self._embedder: Optional[HFEmbeddingModel] = None
        self._index: Optional[FAISSIndex] = None

    def _get_embedder(self) -> HFEmbeddingModel:
        """Lazy-load the embedding model."""
        if self._embedder is None:
            embed_cfg = self.config.get("embedding_model") or {}
            self._embedder = HFEmbeddingModel(embed_cfg)
        return self._embedder

    def _default_cache_key(self) -> str:
        """Create a deterministic cache key from RAG config."""
        fingerprint = {
            "corpus": self.config.get("corpus"),
            "corpus_template": self.config.get("corpus_template"),
            "corpus_mappings": self.config.get("corpus_mappings"),
            "embedding_model": self.config.get("embedding_model"),
        }
        raw = json.dumps(fingerprint, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()[:12]

    def build_or_load_index(
        self,
        documents: List[str],
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> None:
        """Build or load a FAISS index for the provided documents."""
        embedder = self._get_embedder()
        key = cache_key or (self._default_cache_key() if cache_dir else None)
        self._index = FAISSIndex.build_or_load(
            documents=list(documents),
            embed_fn=embedder.embed_texts,
            cache_dir=cache_dir,
            cache_key=key,
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

    def format_context(
        self,
        retrieved: Iterable[Mapping[str, Any]],
        template: str,
        separator: str,
    ) -> str:
        """Format retrieved items into a single context string."""
        rendered: List[str] = []
        for item in retrieved:
            rendered.append(template.format_map(item))
        return separator.join(rendered)
