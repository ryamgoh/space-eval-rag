from __future__ import annotations

from typing import Any, Dict

from inference.model.base_model import BaseModel


class APIModel(BaseModel):
    """Adapter placeholder for API-backed models."""
    def __init__(self, config: Dict[str, Any]):
        """Store API configuration without initializing a client."""
        name = config["name"]
        super().__init__(name=name, model_type="api", max_batch_size=config.get("max_batch_size"))
        self.config = config

    async def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses via an external API (not implemented)."""
        raise NotImplementedError("API model adapter is not implemented yet.")
