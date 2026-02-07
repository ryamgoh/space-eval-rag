from __future__ import annotations

from pathlib import Path


def read_pdf_file(path: Path) -> str:
    """Extract text from a PDF file."""
    try:
        from pypdf import PdfReader
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PDF support requires pypdf. Install it to ingest PDFs.") from exc

    reader = PdfReader(str(path))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages).strip()
