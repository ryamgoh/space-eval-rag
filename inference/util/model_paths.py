from __future__ import annotations

import os
from typing import Optional


def resolve_model_path(model_path: str, local_dir: Optional[str]) -> tuple[str, Optional[str]]:
    """Decide whether to load from a local dir or download into it."""
    if not local_dir:
        return model_path, None
    os.makedirs(local_dir, exist_ok=True)
    marker_files = ("config.json", "model.safetensors", "pytorch_model.bin")
    # Treat local_dir as a model path only if it already looks populated.
    has_model_files = any(os.path.exists(os.path.join(local_dir, name)) for name in marker_files)
    if has_model_files:
        return local_dir, local_dir
    return model_path, local_dir
