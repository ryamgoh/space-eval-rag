from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Mapping

from inference.datasets.readers.base import (
    JSONL_EXTENSIONS,
    PDF_EXTENSIONS,
    DOCX_EXTENSIONS,
    SUPPORTED_EXTENSIONS,
    TEXT_EXTENSIONS,
    normalize_extensions,
)

from langchain_community.document_loaders import JSONLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


class FileCorpusLoader:
    """Load a dataset from local files using LangChain loaders and splitters."""
    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self.chunk_size = config.get("chunk_size")
        self.chunk_overlap = int(config.get("chunk_overlap", 0))
        self.include_metadata = bool(config.get("include_metadata", True))
        self.extensions = normalize_extensions(config.get("extensions"), SUPPORTED_EXTENSIONS)
        self.recursive = bool(config.get("recursive", True))
        self.json_content_key = str(config.get("json_content_key", "text"))
        self.json_jq_schema = config.get("json_jq_schema")

        if self.chunk_size is not None:
            chunk_size = int(self.chunk_size)
            if chunk_size <= 0:
                self.chunk_size = None
            elif self.chunk_overlap >= chunk_size:
                raise ValueError("chunk_overlap must be smaller than chunk_size.")

    def load(self) -> List[Dict[str, Any]]:
        """Load and normalize documents from the configured corpus paths."""
        paths = self._iter_paths()
        documents: List[Dict[str, Any]] = []
        for path in paths:
            documents.extend(self._load_path(path))
        if not documents:
            raise ValueError("No documents loaded from file corpus.")
        return documents

    def _iter_paths(self) -> List[Path]:
        paths: List[Path] = []
        if "paths" in self.config:
            paths.extend(Path(p) for p in self.config["paths"])
        if "path" in self.config:
            paths.append(Path(self.config["path"]))
        if "glob" in self.config:
            pattern = str(self.config["glob"])
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
            docs = self._load_jsonl(path)
        elif suffix in TEXT_EXTENSIONS:
            docs = TextLoader(str(path), encoding="utf-8").load()
        elif suffix in PDF_EXTENSIONS:
            docs = PyPDFLoader(str(path)).load()
        elif suffix in DOCX_EXTENSIONS:
            docs = Docx2txtLoader(str(path)).load()
        else:
            docs = []

        if self.chunk_size:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=int(self.chunk_size),
                chunk_overlap=self.chunk_overlap,
                add_start_index=True,
            )
            docs = splitter.split_documents(docs)

        return self._format_docs(docs)

    def _load_jsonl(self, path: Path):
        jq_schema = self.json_jq_schema or f".{self.json_content_key}"
        loader = JSONLoader(
            file_path=str(path),
            jq_schema=jq_schema,
            json_lines=True,
        )
        try:
            return loader.load()
        except Exception as exc:  # pragma: no cover - surface loader errors
            raise ValueError(
                f"Failed to load JSONL corpus at {path}. "
                "Set corpus.json_jq_schema or corpus.json_content_key if needed, "
                "and ensure the 'jq' Python package is installed."
            ) from exc

    def _format_docs(self, docs) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for idx, doc in enumerate(docs):
            metadata = dict(getattr(doc, "metadata", {}) or {})
            if "source" in metadata and "source_path" not in metadata:
                metadata["source_path"] = metadata["source"]
            if "seq_num" in metadata and "line_number" not in metadata:
                metadata["line_number"] = metadata["seq_num"]
            if "start_index" in metadata and "chunk_start" not in metadata:
                start = int(metadata["start_index"])
                metadata["chunk_start"] = start
                metadata["chunk_end"] = start + len(doc.page_content)
            if self.chunk_size:
                metadata.setdefault("chunk_index", idx)
            if not self.include_metadata:
                metadata = {}
            record = {"text": doc.page_content}
            record.update(metadata)
            results.append(record)
        return results


def load_files_dataset(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load a dataset from local files using LangChain loaders/splitters."""
    return FileCorpusLoader(config).load()
