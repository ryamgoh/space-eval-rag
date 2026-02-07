from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import faiss
import numpy as np


class FAISSIndex:
    """FAISS index plus its backing document list."""
    def __init__(self, index: faiss.Index, docs: List[str]):
        self.index = index
        self.docs = docs

    @classmethod
    def build_or_load(
        cls,
        documents: List[str],
        embed_fn: Callable[[List[str]], np.ndarray],
        cache_dir: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> "FAISSIndex":
        """Build or load a FAISS index for the provided documents."""
        if not documents:
            raise ValueError("RAG corpus is empty; cannot build index.")

        index_path, meta_path = _cache_paths(cache_dir, cache_key)
        if index_path and meta_path and index_path.exists() and meta_path.exists():
            index = faiss.read_index(str(index_path))
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            docs = list(metadata.get("docs", []))
            if not docs:
                raise ValueError("Cached RAG metadata is empty.")
            return cls(index=index, docs=docs)

        embeddings = embed_fn(documents)
        if embeddings.size == 0:
            raise ValueError("Failed to embed RAG corpus.")
        dim = embeddings.shape[1]
        # Inner-product index works with normalized embeddings for cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        instance = cls(index=index, docs=list(documents))
        if index_path and meta_path:
            _write_cache(index_path, meta_path, instance.docs, dim, index)
        return instance

    def search(self, query_vec: np.ndarray, k: int) -> List[Tuple[int, float]]:
        """Return (doc_id, score) pairs for a single query vector."""
        if query_vec.size == 0:
            return []
        k = min(int(k), len(self.docs))
        scores, indices = self.index.search(query_vec, k)
        return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx >= 0]


def _cache_paths(
    cache_dir: Optional[str],
    cache_key: Optional[str],
) -> tuple[Path | None, Path | None]:
    if not cache_dir or not cache_key:
        return None, None
    cache_root = Path(cache_dir) / cache_key
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / "index.faiss", cache_root / "metadata.json"


def _write_cache(
    index_path: Path,
    meta_path: Path,
    docs: Iterable[str],
    dim: int,
    index: faiss.Index,
) -> None:
    faiss.write_index(index, str(index_path))
    meta_path.write_text(
        json.dumps({"docs": list(docs), "dim": dim}, indent=2),
        encoding="utf-8",
    )
