from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

from inference.datasets.readers.base import ensure_text_doc, iter_non_empty_lines, with_line_metadata


def read_jsonl_file(path: Path, include_metadata: bool) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    docs: List[Dict[str, Any]] = []
    for line_num, line in enumerate(iter_non_empty_lines(path), start=1):
        obj = ensure_text_doc(json.loads(line))
        if include_metadata:
            obj = with_line_metadata(obj, path, line_num)
        docs.append(obj)
    return docs
