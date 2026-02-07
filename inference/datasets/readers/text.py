from __future__ import annotations

from pathlib import Path


def read_text_file(path: Path) -> str:
    """Read a UTF-8 text/markdown file into a string."""
    return path.read_text(encoding="utf-8")
