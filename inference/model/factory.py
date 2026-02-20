from __future__ import annotations

from inference.config.models import (
    APIModelConfig,
    HuggingFaceModelConfig,
    ModelConfig,
    VLLMModelConfig,
)
from inference.model.api import APIModel
from inference.model.core_model import CoreModelInterface
from inference.model.huggingface import HuggingFaceModel
from inference.model.vllm import VLLMModel


class ModelFactory:
    """Factory for creating model adapters from config."""

    @staticmethod
    def get_model(model_config: ModelConfig) -> CoreModelInterface:
        """Instantiate the appropriate model adapter based on config type."""
        if isinstance(model_config, HuggingFaceModelConfig):
            return HuggingFaceModel(model_config)
        if isinstance(model_config, VLLMModelConfig):
            return VLLMModel(model_config)
        if isinstance(model_config, APIModelConfig):
            return APIModel(model_config)
        raise ValueError(f"Unknown model type: {type(model_config)}")
