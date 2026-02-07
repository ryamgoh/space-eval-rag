from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from inference.datasets.readers import read_jsonl_file, read_pdf_file, read_text_file
from inference.datasets.readers.base import (
    JSONL_EXTENSIONS,
    PDF_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    TEXT_EXTENSIONS,
    chunk_text,
    normalize_extensions,
)

class FileCorpusLoader:
    """Load a dataset from local files in txt, md, jsonl, or pdf formats."""
    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.chunk_size = config.get("chunk_size")
        self.chunk_overlap = int(config.get("chunk_overlap", 0))
        self.include_metadata = bool(config.get("include_metadata", True))
        self.extensions = normalize_extensions(config.get("extensions"), SUPPORTED_EXTENSIONS)
        self.recursive = bool(config.get("recursive", True))

    def load(self) -> List[Dict[str, Any]]:
        """Load and normalize documents from the configured corpus paths."""
        paths = self._iter_paths()
        docs: List[Dict[str, Any]] = []
        for path in paths:
            docs.extend(self._load_path(path))
        if not docs:
            raise ValueError("No documents loaded from file corpus.")
        return docs

    def _iter_paths(self) -> List[Path]:
        paths: List[Path] = []
        if "paths" in self.config:
            paths.extend(Path(p) for p in self.config["paths"])
        if "path" in self.config:
            paths.append(Path(self.config["path"]))
        if "glob" in self.config:
            pattern = str(self.config["glob"])
            # Use glob for patterns like data/**/*.jsonl.
            paths.extend(Path(p) for p in glob.glob(pattern, recursive=self.recursive))

        if not paths:
            raise ValueError("files corpus requires 'path', 'paths', or 'glob'.")

        collected: List[Path] = []
        for path in paths:
            if path.is_dir():
                for root, _, files in os.walk(path):
                    root_path = Path(root)
                    if not self.recursive and root_path != path:
                        continue
                    for name in files:
                        candidate = root_path / name
                        if candidate.suffix.lower() in self.extensions:
                            collected.append(candidate)
            elif path.is_file():
                if path.suffix.lower() in self.extensions:
                    collected.append(path)
            else:
                raise ValueError(f"Corpus path does not exist: {path}")

        return sorted(set(collected))

    def _load_path(self, path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix in JSONL_EXTENSIONS:
            return self._load_jsonl(path)
        if suffix in TEXT_EXTENSIONS:
            text = read_text_file(path)
            return list(self._emit_text_docs(text, self._base_doc(path)))
        if suffix in PDF_EXTENSIONS:
            text = read_pdf_file(path)
            return list(self._emit_text_docs(text, self._base_doc(path)))
        return []

    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        for obj in read_jsonl_file(path, include_metadata=self.include_metadata):
            docs.extend(self._emit_text_docs(obj.get("text"), obj))
        return docs

    def _base_doc(self, path: Path) -> Dict[str, Any]:
        if not self.include_metadata:
            return {}
        return {"source_path": str(path)}

    def _emit_text_docs(self, text: str | None, base_doc: Dict[str, Any]) -> Iterable[Dict[str, Any]]:
        if not text:
            return []
        if not self.chunk_size:
            doc = dict(base_doc)
            doc["text"] = text
            return [doc]

        docs: List[Dict[str, Any]] = []
        for chunk, start, end, idx in self._chunk_text(text):
            doc = dict(base_doc)
            doc["text"] = chunk
            doc["chunk_index"] = idx
            doc["chunk_start"] = start
            doc["chunk_end"] = end
            docs.append(doc)
        return docs

    def _chunk_text(self, text: str) -> Iterable[Tuple[str, int, int, int]]:
        return chunk_text(text, int(self.chunk_size), self.chunk_overlap)

def load_files_dataset(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load a dataset from local files in txt, md, jsonl, or pdf formats."""
    return FileCorpusLoader(config).load()
