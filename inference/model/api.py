from __future__ import annotations

from inference.config.models import APIModelConfig
from inference.model.core_model import CoreModelInterface


class APIModel(CoreModelInterface):
    """Adapter placeholder for API-backed models."""

    def __init__(self, config: APIModelConfig):
        """Store API configuration without initializing a client."""
        super().__init__(
            name=config.name, model_type="api", max_batch_size=config.max_batch_size
        )
        self.config = config

    async def generate_batch(self, prompts: list[str], **kwargs) -> list[str]:
        """Generate responses via an external API (not implemented)."""
        raise NotImplementedError("API model adapter is not implemented yet.")
