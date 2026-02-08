from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Sequence

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
DOCX_EXTENSIONS = {".docx"}
JSONL_EXTENSIONS = {".jsonl"}
SUPPORTED_EXTENSIONS = set().union(TEXT_EXTENSIONS, PDF_EXTENSIONS, DOCX_EXTENSIONS, JSONL_EXTENSIONS)


def iter_non_empty_lines(path: Path) -> Iterator[str]:
    """Yield non-empty lines from a text file."""
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield stripped


def with_line_metadata(doc: Dict[str, Any], path: Path, line_number: int) -> Dict[str, Any]:
    """Attach source metadata to a JSONL row."""
    enriched = dict(doc)
    enriched["source_path"] = str(path)
    enriched["line_number"] = line_number
    return enriched


def ensure_text_doc(value: Any) -> Dict[str, Any]:
    """Normalize a JSONL row into a dict with at least a text field."""
    if isinstance(value, dict):
        return value
    return {"text": str(value)}


def chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> Iterable[tuple[str, int, int, int]]:
    """Yield (chunk, start, end, index) for a text string."""
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


def normalize_extensions(extensions: Sequence[str] | None, defaults: Iterable[str]) -> set[str]:
    """Normalize extensions (accepts with/without leading dots)."""
    if not extensions:
        return set(defaults)
    normalized: set[str] = set()
    for ext in extensions:
        if not ext:
            continue
        ext_str = ext if ext.startswith(".") else f".{ext}"
        normalized.add(ext_str.lower())
    return normalized
