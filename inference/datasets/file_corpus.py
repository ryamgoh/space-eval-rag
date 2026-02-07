from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


SUPPORTED_EXTENSIONS = {".txt", ".md", ".jsonl", ".pdf"}


def _normalize_extensions(extensions: Sequence[str] | None) -> set[str]:
    if not extensions:
        return set(SUPPORTED_EXTENSIONS)
    normalized: set[str] = set()
    for ext in extensions:
        if not ext:
            continue
        ext_str = ext if ext.startswith(".") else f".{ext}"
        normalized.add(ext_str.lower())
    return normalized


def _iter_paths(config: Mapping[str, Any]) -> List[Path]:
    paths: List[Path] = []
    if "paths" in config:
        paths.extend(Path(p) for p in config["paths"])
    if "path" in config:
        paths.append(Path(config["path"]))
    if "glob" in config:
        pattern = str(config["glob"])
        recursive = bool(config.get("recursive", True))
        # Use glob for patterns like data/**/*.jsonl.
        paths.extend(Path(p) for p in glob.glob(pattern, recursive=recursive))

    if not paths:
        raise ValueError("files corpus requires 'path', 'paths', or 'glob'.")

    include_exts = _normalize_extensions(config.get("extensions"))
    recursive_dirs = bool(config.get("recursive", True))
    collected: List[Path] = []
    for path in paths:
        if path.is_dir():
            for root, _, files in os.walk(path):
                root_path = Path(root)
                if not recursive_dirs and root_path != path:
                    continue
                for name in files:
                    candidate = root_path / name
                    if candidate.suffix.lower() in include_exts:
                        collected.append(candidate)
        elif path.is_file():
            if path.suffix.lower() in include_exts:
                collected.append(path)
        else:
            raise ValueError(f"Corpus path does not exist: {path}")

    return sorted(set(collected))


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> Iterable[tuple[str, int, int, int]]:
    if chunk_size <= 0:
        yield text, 0, len(text), 0
        return
    step = chunk_size - chunk_overlap
    if step <= 0:
        raise ValueError("chunk_overlap must be smaller than chunk_size.")
    chunk_index = 0
    for start in range(0, len(text), step):
        end = min(start + chunk_size, len(text))
        chunk = text[start:end]
        if chunk:
            yield chunk, start, end, chunk_index
            chunk_index += 1


def _read_pdf(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PDF support requires pypdf. Install it to ingest PDFs.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _emit_text_docs(
    text: str,
    base_doc: Dict[str, Any],
    chunk_size: int | None,
    chunk_overlap: int,
) -> Iterable[Dict[str, Any]]:
    if not text:
        return
    if not chunk_size:
        # Single doc with the full text.
        doc = dict(base_doc)
        doc["text"] = text
        yield doc
        return

    for chunk, start, end, idx in _chunk_text(text, chunk_size, chunk_overlap):
        # Emit chunk-level documents with offsets for traceability.
        doc = dict(base_doc)
        doc["text"] = chunk
        doc["chunk_index"] = idx
        doc["chunk_start"] = start
        doc["chunk_end"] = end
        yield doc


def load_files_dataset(config: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Load a dataset from local files in txt, md, jsonl, or pdf formats."""
    paths = _iter_paths(config)
    chunk_size = config.get("chunk_size")
    chunk_overlap = int(config.get("chunk_overlap", 0))
    include_metadata = bool(config.get("include_metadata", True))
    docs: List[Dict[str, Any]] = []

    for path in paths:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for line_num, line in enumerate(handle, start=1):
                    stripped = line.strip()
                    if not stripped:
                        continue
                    # Accept dict rows or coerce primitives to {"text": "..."}.
                    obj = json.loads(stripped)
                    if not isinstance(obj, dict):
                        obj = {"text": str(obj)}
                    if include_metadata:
                        obj = dict(obj)
                        obj["source_path"] = str(path)
                        obj["line_number"] = line_num
                    if chunk_size and isinstance(obj.get("text"), str):
                        for chunk_doc in _emit_text_docs(
                            obj["text"], obj, int(chunk_size), chunk_overlap
                        ):
                            docs.append(chunk_doc)
                    else:
                        docs.append(obj)
            continue

        if suffix in {".txt", ".md"}:
            text = _read_text(path)
        elif suffix == ".pdf":
            text = _read_pdf(path)
        else:
            continue

        base_doc: Dict[str, Any] = {}
        if include_metadata:
            base_doc["source_path"] = str(path)
        for doc in _emit_text_docs(text, base_doc, int(chunk_size) if chunk_size else None, chunk_overlap):
            docs.append(doc)

    if not docs:
        raise ValueError("No documents loaded from file corpus.")
    return docs
