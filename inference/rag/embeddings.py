from __future__ import annotations

from typing import Any, Iterable, List, Mapping

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from inference.util.model_paths import resolve_model_path


class HFEmbeddingModel:
    """Simple Hugging Face embedding helper."""
    def __init__(self, config: Mapping[str, Any]):
        """Load tokenizer/model for embedding inference."""
        model_path = config["model_path"]
        local_dir = config.get("local_dir")
        resolved_path, cache_dir = resolve_model_path(model_path, local_dir)
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
