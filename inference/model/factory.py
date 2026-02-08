from __future__ import annotations

from typing import Any, Dict

from inference.model.api import APIModel
from inference.model.base_model import BaseModel
from inference.model.huggingface import HuggingFaceModel
from inference.model.vllm import VLLMModel


class ModelFactory:
    """Factory for creating model adapters from config."""
    @staticmethod
    def get_model(model_config: Dict[str, Any]) -> BaseModel:
        """Instantiate the appropriate model adapter based on config type."""
        model_type = model_config["type"].lower()
        if model_type == "huggingface":
            return HuggingFaceModel(model_config)
        if model_type == "vllm":
            return VLLMModel(model_config)
        if model_type == "api":
            return APIModel(model_config)
        raise ValueError(f"Unknown model type: {model_type}")
