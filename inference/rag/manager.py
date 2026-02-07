from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


def _resolve_model_path(model_path: str, local_dir: Optional[str]) -> tuple[str, Optional[str]]:
    """Decide whether to load from a local dir or download into it."""
    if not local_dir:
        return model_path, None
    os.makedirs(local_dir, exist_ok=True)
    marker_files = ("config.json", "model.safetensors", "pytorch_model.bin")
    has_model_files = any(os.path.exists(os.path.join(local_dir, name)) for name in marker_files)
    if has_model_files:
        return local_dir, local_dir
    return model_path, local_dir


class HFEmbeddingModel:
    """Simple Hugging Face embedding helper."""
    def __init__(self, config: Mapping[str, Any]):
        """Load tokenizer/model for embedding inference."""
        model_path = config["model_path"]
        local_dir = config.get("local_dir")
        resolved_path, cache_dir = _resolve_model_path(model_path, local_dir)
        trust_remote = config.get("trust_remote_code", False)
        self.max_length = int(config.get("max_length", 256))
        self.batch_size = int(config.get("batch_size", 32))
        self.pooling = str(config.get("pooling", "mean")).lower()
        self.device = config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            resolved_path, cache_dir=cache_dir, trust_remote_code=trust_remote
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.unk_token
        self.model = AutoModel.from_pretrained(
            resolved_path, cache_dir=cache_dir, trust_remote_code=trust_remote
        )
        self.model.to(self.device)
        self.model.eval()

    def embed_texts(self, texts: Iterable[str]) -> np.ndarray:
        """Embed texts into a dense float32 numpy array."""
        text_list = list(texts)
        if not text_list:
            return np.zeros((0, 0), dtype=np.float32)
        embeddings: List[np.ndarray] = []
        for start in range(0, len(text_list), self.batch_size):
            batch = text_list[start : start + self.batch_size]
            encoded = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.no_grad():
                outputs = self.model(**encoded)
            hidden_state = outputs.last_hidden_state
            if self.pooling == "cls":
                # CLS pooling keeps the first token representation.
                pooled = hidden_state[:, 0]
            else:
                # Mean pooling over non-padding tokens.
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden_state.size()).float()
                summed = (hidden_state * mask).sum(dim=1)
                counts = mask.sum(dim=1).clamp(min=1e-9)
                pooled = summed / counts
            # Normalize for cosine similarity with inner product search.
            normalized = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(normalized.cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)


class RAGManager:
    """Retrieval-augmented generation manager using FAISS."""
    def __init__(self, config: Mapping[str, Any]):
        """Initialize RAG configuration without building an index yet."""
        self.config = dict(config)
        self._embedder: Optional[HFEmbeddingModel] = None
        self._index: Optional[faiss.Index] = None
        self._docs: List[str] = []

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
        self._docs = list(documents)
        if not self._docs:
            raise ValueError("RAG corpus is empty; cannot build index.")

        index_path = None
        meta_path = None
        if cache_dir:
            key = cache_key or self._default_cache_key()
            cache_root = Path(cache_dir) / key
            cache_root.mkdir(parents=True, exist_ok=True)
            index_path = cache_root / "index.faiss"
            meta_path = cache_root / "metadata.json"
            if index_path.exists() and meta_path.exists():
                # Reuse cached index + doc list when present.
                self._index = faiss.read_index(str(index_path))
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                self._docs = list(metadata.get("docs", []))
                if not self._docs:
                    raise ValueError("Cached RAG metadata is empty.")
                return

        embedder = self._get_embedder()
        embeddings = embedder.embed_texts(self._docs)
        if embeddings.size == 0:
            raise ValueError("Failed to embed RAG corpus.")
        dim = embeddings.shape[1]
        # Inner-product index works with normalized embeddings for cosine similarity.
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        self._index = index

        if index_path and meta_path:
            faiss.write_index(index, str(index_path))
            meta_path.write_text(
                json.dumps({"docs": self._docs, "dim": dim}, indent=2),
                encoding="utf-8",
            )

    def retrieve(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Retrieve top-k documents for a query."""
        if not query.strip():
            return []
        if self._index is None:
            raise RuntimeError("RAG index is not initialized.")
        embedder = self._get_embedder()
        query_vec = embedder.embed_texts([query])
        if query_vec.size == 0:
            return []
        k = min(int(k), len(self._docs))
        scores, indices = self._index.search(query_vec, k)
        results: List[Dict[str, Any]] = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0:
                continue
            results.append(
                {"id": int(idx), "rank": rank, "score": float(score), "text": self._docs[idx]}
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
